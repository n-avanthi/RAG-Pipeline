"""
rag_pipeline.py
---------------
Online pipeline — Step 4 of AC-RAG.

At query time:
  1. route_query()                  — LLM reads cluster profiles + geometric
                                      centroid ranking, picks 2-3 clusters
  2. retrieve_with_hnsw_filtered()  — HNSW search filtered to selected clusters + MMR
  3. assemble_rag_context()         — interleaves chunks into a context string
  4. generate_answer()              — final answer over assembled context

Change from previous version:
  - Cluster centroids are precomputed and cached at load_pipeline_state() time.
  - route_query() computes cosine similarity from the query vector to each
    centroid and injects a geometric ranking into the router prompt as a
    tiebreaker alongside the LLM text profiles. This grounds routing in
    actual embedding geometry rather than purely LLM-inferred semantics.

LLM backend: OpenAI (gpt-4o-mini)
Ollama calls are commented out below each OpenAI equivalent.
"""

import os
import re
import json
import time
import logging
import textwrap

import numpy as np
import torch
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import (
    MODEL_NAME, DEVICE, EMBEDDING_DIM,
    CLUSTER_PROFILES_PATH, ENRICHED_METADATA_PATH,
    INDEX_PATH, EMBEDDINGS_PATH,
    OPENAI_API_KEY, LLM_MODEL_NAME,
    ROUTER_TEMPERATURE, ROUTER_MAX_RETRIES,
    ROUTER_MIN_CLUSTERS, ROUTER_MAX_CLUSTERS,
    RETRIEVAL_TOP_K_PER_CLUSTER, RETRIEVAL_DIVERSITY_LAMBDA,
    RANDOM_SEED,
)
from app.preprocess import get_bge_model, get_hnsw_index, load_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client (module-level singleton)
# ---------------------------------------------------------------------------
_openai_client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it in your environment before starting the server."
            )
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# ---------------------------------------------------------------------------
# Ollama client — commented out
# ---------------------------------------------------------------------------
# import requests
# OLLAMA_HOST    = "http://localhost:11434"
# ROUTER_TIMEOUT_S   = 60
# ROUTER_NUM_PREDICT = 256

# ---------------------------------------------------------------------------
# Pipeline state — loaded once at startup
# ---------------------------------------------------------------------------

_cluster_profiles    : dict              = {}
_enriched            : list[dict]        = []
_embeddings_matrix   : np.ndarray | None = None

_cluster_to_faiss_ids: dict[int, set[int]] = {}
_index_to_record     : dict[int, dict]     = {}
_valid_cluster_ids   : list[int]           = []

# Precomputed unit-normalised centroids: {cluster_id (int) -> np.ndarray (1024,)}
_cluster_centroids   : dict[int, np.ndarray] = {}


def _precompute_centroids() -> None:
    """
    Computes and caches a unit-normalised centroid vector for every cluster.
    Called once inside load_pipeline_state() after embeddings are loaded.
    Cheap: one mean + one norm per cluster, done at startup not per query.
    """
    global _cluster_centroids
    _cluster_centroids.clear()

    for cid, faiss_ids in _cluster_to_faiss_ids.items():
        indices = list(faiss_ids)
        vecs    = _embeddings_matrix[indices]          # (n, 1024)
        centroid = vecs.mean(axis=0)                   # (1024,)
        norm     = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
        _cluster_centroids[cid] = centroid.astype("float32")

    logger.info("Centroids precomputed for %d clusters.", len(_cluster_centroids))


def load_pipeline_state() -> bool:
    global _cluster_profiles, _enriched, _embeddings_matrix
    global _cluster_to_faiss_ids, _index_to_record, _valid_cluster_ids

    required = [CLUSTER_PROFILES_PATH, ENRICHED_METADATA_PATH,
                INDEX_PATH, EMBEDDINGS_PATH]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        logger.warning("Pipeline artefacts missing — cannot serve queries: %s", missing)
        return False

    with open(CLUSTER_PROFILES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _cluster_profiles = data.get("profiles", {})

    _enriched          = load_json(ENRICHED_METADATA_PATH)
    _embeddings_matrix = np.load(EMBEDDINGS_PATH).astype("float32")

    _cluster_to_faiss_ids.clear()
    _index_to_record.clear()

    for record in _enriched:
        cid  = record["primary_cluster"]
        fidx = record["index"]
        _cluster_to_faiss_ids.setdefault(cid, set()).add(fidx)
        _index_to_record[fidx] = record

    _valid_cluster_ids = sorted(map(int, _cluster_profiles.keys()))

    # Precompute centroids now that embeddings + cluster membership are loaded
    _precompute_centroids()

    get_bge_model()
    get_hnsw_index()

    logger.info("Pipeline state loaded — %d clusters, %d chunks",
                len(_cluster_profiles), len(_enriched))
    return True


def is_pipeline_ready() -> bool:
    return (
        bool(_cluster_profiles)
        and bool(_enriched)
        and _embeddings_matrix is not None
    )


# ---------------------------------------------------------------------------
# Router prompt helpers
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = textwrap.dedent("""
    You are a strategic retrieval agent for a document corpus.

    Your role is to analyze the user query and select the most
    relevant clusters from the provided Cluster Profiles.

    DIVERSITY MANDATE
    -----------------
    - Select between 2 and 3 clusters that offer CONTRASTING
      but relevant perspectives to fully answer the query.
    - You MUST prioritize minority viewpoints (marked
      [MINORITY VIEW]) to avoid dominant document bias.
    - Never select clusters whose 'Distinct edge' fields
      describe the same narrow topic.
    - Prefer breadth: one cluster may address causes, another
      effects, another a different stakeholder perspective.

    GEOMETRIC SIGNAL
    ----------------
    - A ranked list of clusters by embedding-space proximity
      to the query is provided (query→centroid cosine similarity).
    - Use this as a tiebreaker when text profiles are ambiguous
      or when two clusters seem equally relevant by description.
    - Do not follow it blindly — a geometrically close cluster
      may still be thematically redundant with another selection.

    OUTPUT FORMAT
    -------------
    Return ONLY a JSON object matching this exact schema:
    {
      "selected_clusters": [id1, id2],
      "reasoning": "One sentence explaining why these clusters
                    together best answer the query."
    }

    No markdown. No explanation outside the JSON object.
    The 'selected_clusters' field must be a list of integers.
""").strip()


def _format_profiles_for_llm(q_vec: np.ndarray | None = None) -> str:
    """
    Formats cluster profiles for the router prompt.
    If q_vec is supplied, appends a geometric ranking section
    (cosine similarity from query to each precomputed centroid).
    """
    counts       = [e["n_members"] for e in _cluster_profiles.values()]
    median_count = float(np.median(counts))
    lines        = ["CLUSTER PROFILES", "=" * 60]

    for cid_str in sorted(_cluster_profiles.keys(), key=int):
        entry    = _cluster_profiles[cid_str]
        profile  = entry["profile"]
        n        = entry["n_members"]
        avg_p    = entry["avg_prob"]
        minority = " [MINORITY VIEW]" if n < median_count else ""

        lines += [
            f"\nCluster {cid_str}{minority}",
            f"  Members       : {n} chunks (avg confidence: {avg_p:.3f})",
            f"  Theme         : {profile['primary_theme']}",
            f"  Key entities  : {' | '.join(profile['key_entities'])}",
            f"  Distinct edge : {profile['contrastive_edge'][:150]}",
        ]

    lines += ["", "=" * 60]

    # Inject geometric ranking if query vector is available
    if q_vec is not None and _cluster_centroids:
        sims = {
            cid: float(_cluster_centroids[cid] @ q_vec)
            for cid in _cluster_centroids
        }
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        lines += [
            "",
            "GEOMETRIC RANKING  (query → cluster centroid cosine similarity)",
            "-" * 60,
        ]
        for rank, (cid, sim) in enumerate(ranked, 1):
            theme = _cluster_profiles.get(str(cid), {}).get(
                "profile", {}).get("primary_theme", f"Cluster {cid}")
            lines.append(f"  #{rank:>2}  Cluster {cid:>2}  sim={sim:.4f}  {theme}")
        lines += [
            "",
            "Use this as a tiebreaker when text profiles are ambiguous.",
            "=" * 60,
        ]

    return "\n".join(lines)


def _build_router_prompt(user_query: str, q_vec: np.ndarray | None = None) -> str:
    profiles_context = _format_profiles_for_llm(q_vec)
    id_list          = ", ".join(str(cid) for cid in _valid_cluster_ids)

    return textwrap.dedent(f"""
        {profiles_context}

        USER QUERY
        ----------
        {user_query}

        VALID CLUSTER IDs: [{id_list}]

        Remember: return ONLY valid JSON matching the schema above.
        selected_clusters must contain integers from the valid list.
    """).strip()


def _extract_router_json(raw: str) -> dict:
    raw = raw.strip()

    def validate(obj: dict) -> dict:
        if "selected_clusters" not in obj:
            raise ValueError("Missing 'selected_clusters' key.")
        ids = [int(i) for i in obj["selected_clusters"]]
        seen, deduped = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        ids     = deduped
        invalid = [i for i in ids if i not in _valid_cluster_ids]
        if invalid:
            raise ValueError(f"Invalid cluster ids: {invalid}")
        if len(ids) < ROUTER_MIN_CLUSTERS:
            raise ValueError(f"Too few clusters: {len(ids)} (min={ROUTER_MIN_CLUSTERS})")
        obj["selected_clusters"] = ids[:ROUTER_MAX_CLUSTERS]
        obj.setdefault("reasoning", "")
        return obj

    try:
        return validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    brace_m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if brace_m:
        try:
            return validate(json.loads(brace_m.group()))
        except (json.JSONDecodeError, ValueError):
            pass

    arr_m = re.search(r'"selected_clusters"\s*:\s*\[([^\]]+)\]', raw)
    if arr_m:
        try:
            raw_ids  = [
                int(x.strip())
                for x in arr_m.group(1).split(",")
                if x.strip().lstrip("-").isdigit()
            ]
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
            return validate({
                "selected_clusters": raw_ids,
                "reasoning": reason_m.group(1) if reason_m else "",
            })
        except (ValueError, AttributeError):
            pass

    raise ValueError(f"All JSON extraction strategies failed. Raw: {raw[:400]}")


# ---------------------------------------------------------------------------
# Route query — OpenAI
# ---------------------------------------------------------------------------

def route_query(user_query: str) -> dict:
    """
    Routes the query to 2-3 clusters via OpenAI.
    Embeds the query first so that centroid proximity can be injected
    into the router prompt as a geometric tiebreaker.
    Returns dict: selected_clusters, reasoning, cluster_themes, llm_meta.
    """
    # Embed the query once — reused for both routing and retrieval
    model = get_bge_model()
    with torch.no_grad():
        q_vec = model.encode(
            [user_query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
            show_progress_bar    = False,
        ).astype("float32")[0]

    client = get_openai_client()
    prompt = _build_router_prompt(user_query, q_vec)
    logger.info("Router prompt length: %d chars", len(prompt))

    last_error = None

    for attempt in range(1, ROUTER_MAX_RETRIES + 1):
        current_prompt = prompt
        if attempt > 1:
            current_prompt += (
                "\n\nCRITICAL: Return ONLY a JSON object. "
                "selected_clusters must be integers. No prose."
            )

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model       = LLM_MODEL_NAME,
                temperature = ROUTER_TEMPERATURE,
                messages    = [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": current_prompt},
                ],
                response_format = {"type": "json_object"},
            )
        except Exception as e:
            last_error = str(e)
            logger.warning("OpenAI router call failed (attempt %d): %s", attempt, e)
            time.sleep(1)
            continue

        elapsed  = time.perf_counter() - t0
        raw_text = response.choices[0].message.content.strip()

        try:
            parsed   = _extract_router_json(raw_text)
            selected = parsed["selected_clusters"]
            themes   = {
                cid: _cluster_profiles[str(cid)]["profile"]["primary_theme"]
                for cid in selected
            }

            # Attach centroid sims for the selected clusters (useful for logging/debug)
            centroid_sims = {
                cid: round(float(_cluster_centroids[cid] @ q_vec), 4)
                for cid in selected
                if cid in _cluster_centroids
            }

            logger.info("Router selected clusters %s in %.2fs (centroid sims: %s)",
                        selected, elapsed, centroid_sims)
            return {
                "query"            : user_query,
                "q_vec"            : q_vec,          # carried forward to retrieval
                "selected_clusters": selected,
                "reasoning"        : parsed.get("reasoning", ""),
                "cluster_themes"   : {str(k): v for k, v in themes.items()},
                "centroid_sims"    : {str(k): v for k, v in centroid_sims.items()},
                "llm_meta"         : {
                    "attempt"          : attempt,
                    "elapsed_s"        : round(elapsed, 2),
                    "prompt_tokens"    : response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        except ValueError as e:
            last_error = str(e)
            time.sleep(1)

    raise RuntimeError(
        f"Query routing failed after {ROUTER_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )

# ---------------------------------------------------------------------------
# Ollama route_query — commented out
# ---------------------------------------------------------------------------
# def route_query_ollama(user_query: str) -> dict:
#     payload = {
#         "model": LLM_MODEL_NAME, "prompt": prompt, "stream": False,
#         "format": "json",
#         "options": {"temperature": ROUTER_TEMPERATURE, "num_predict": ROUTER_NUM_PREDICT},
#     }
#     r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=ROUTER_TIMEOUT_S)
#     ...


# ---------------------------------------------------------------------------
# HNSW retrieval + MMR
# ---------------------------------------------------------------------------

def _mmr_select(
    q_vec     : np.ndarray,
    candidates: list[tuple[float, int]],
    top_k     : int,
    lambda_   : float,
) -> list[tuple[float, int]]:
    selected  = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x[0])
            selected.append(best)
            remaining.remove(best)
            continue

        best_mmr, best_cand = -float("inf"), None
        for cand in remaining:
            sim_to_query    = cand[0]
            sim_to_selected = max(
                1.0 - abs(cand[0] - sel[0]) for sel in selected
            )
            mmr = lambda_ * sim_to_query - (1 - lambda_) * sim_to_selected
            if mmr > best_mmr:
                best_mmr  = mmr
                best_cand = cand

        if best_cand:
            selected.append(best_cand)
            remaining.remove(best_cand)

    return selected


def retrieve_with_hnsw_filtered(
    routing_result  : dict,
    top_k           : int   = RETRIEVAL_TOP_K_PER_CLUSTER,
    diversity_lambda: float  = RETRIEVAL_DIVERSITY_LAMBDA,
) -> dict:
    """
    Retrieves chunks from the HNSW index filtered to the router-selected clusters.
    Reuses the query vector already computed in route_query() if present,
    avoiding a redundant BGE-M3 encode call.
    """
    selected = routing_result["selected_clusters"]
    query    = routing_result["query"]

    # Reuse q_vec from routing if available (avoids double encoding)
    if "q_vec" in routing_result:
        q_vec = routing_result["q_vec"].reshape(1, -1).astype("float32")
    else:
        model = get_bge_model()
        with torch.no_grad():
            q_vec = model.encode(
                [query],
                convert_to_numpy     = True,
                normalize_embeddings = True,
                show_progress_bar    = False,
            ).astype("float32")

    hnsw_index = get_hnsw_index()
    if hnsw_index is None:
        raise RuntimeError("HNSW index not loaded. Run preprocessing first.")

    oversample           = max(top_k * 10, 50)
    distances, indices   = hnsw_index.search(q_vec, oversample)

    global_results = [
        (float(distances[0][i]), int(indices[0][i]))
        for i in range(len(indices[0]))
        if indices[0][i] != -1
    ]

    retrieved = {}
    for cid in selected:
        valid_ids    = _cluster_to_faiss_ids.get(cid, set())
        cluster_hits = [
            (score, fidx) for score, fidx in global_results if fidx in valid_ids
        ]

        if not cluster_hits:
            fallback = [r for r in _enriched if r["primary_cluster"] == cid]
            fallback.sort(key=lambda r: r["primary_probability"], reverse=True)
            retrieved[cid] = fallback[:top_k]
            continue

        if diversity_lambda == 0.0 or len(cluster_hits) <= top_k:
            selected_hits = cluster_hits[:top_k]
        else:
            selected_hits = _mmr_select(q_vec[0], cluster_hits, top_k, diversity_lambda)

        chunks_out = []
        for score, fidx in selected_hits:
            record = dict(_index_to_record[fidx])
            record["similarity_score"] = round(score, 6)
            chunks_out.append(record)

        retrieved[cid] = chunks_out

    return retrieved


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_rag_context(retrieved: dict, routing: dict, max_chars: int = 3000) -> str:
    lines    = ["RETRIEVED CONTEXT", "=" * 55]
    clusters = list(retrieved.keys())
    max_len  = max(len(v) for v in retrieved.values())
    total    = 0

    for slot in range(max_len):
        for cid in clusters:
            chunks = retrieved[cid]
            if slot >= len(chunks):
                continue
            chunk  = chunks[slot]
            theme  = routing["cluster_themes"].get(str(cid), f"Cluster {cid}")
            header = f"\n[Cluster {cid} | {theme} | chunk: {chunk['chunk_id']}]"
            body   = chunk["text"].strip()
            block  = f"{header}\n{body}"

            if total + len(block) > max_chars:
                lines.append(f"\n… [context truncated at {max_chars} chars]")
                return "\n".join(lines)

            lines.append(block)
            total += len(block)

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer generation — OpenAI
# ---------------------------------------------------------------------------

def generate_answer(user_query: str, rag_context: str) -> tuple[str, float]:
    """
    Passes the assembled context to OpenAI for a final answer.
    Returns (answer_text, elapsed_seconds).
    """
    client = get_openai_client()

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model       = LLM_MODEL_NAME,
        temperature = 0.3,
        messages    = [
            {
                "role"   : "system",
                "content": (
                    "You are a knowledgeable analyst. Answer the user's question "
                    "using ONLY the provided context. Be specific, cite facts from "
                    "the context, and keep your answer to 4-6 sentences."
                ),
            },
            {
                "role"   : "user",
                "content": f"CONTEXT:\n{rag_context}\n\nQUESTION: {user_query}",
            },
        ],
        max_tokens = 400,
    )
    elapsed = time.perf_counter() - t0
    answer  = response.choices[0].message.content.strip()
    logger.info("Answer generated in %.2fs (%d tokens)",
                elapsed, response.usage.completion_tokens)
    return answer, round(elapsed, 2)

# ---------------------------------------------------------------------------
# Ollama generate_answer — commented out
# ---------------------------------------------------------------------------
# def generate_answer_ollama(user_query: str, rag_context: str) -> tuple[str, float]:
#     prompt = f"...{rag_context}...\nQUESTION: {user_query}\nANSWER:"
#     r = requests.post(f"{OLLAMA_HOST}/api/generate",
#                       json={"model": LLM_MODEL_NAME, "prompt": prompt,
#                             "stream": False, "options": {"temperature": 0.3, "num_predict": 300}},
#                       timeout=120)
#     ...


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------

def check_ollama_alive() -> bool:
    # Ollama no longer used — always returns False
    # To restore: import requests; r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    return False


def check_openai_alive() -> bool:
    """Lightweight check — just verifies the API key is set."""
    return bool(OPENAI_API_KEY)