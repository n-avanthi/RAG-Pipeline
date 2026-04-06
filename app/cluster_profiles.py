"""
cluster_profiles.py
-------------------
Offline pipeline — Steps 2 & 3 of AC-RAG.

Step 2: Soft Clustering
  fit_umap_5d()               — 1024D → 5D for GMM input only
  bic_sweep()                 — find optimal k
  fit_final_gmm()             — production GMM at best_k
  compute_soft_assignments()  — primary + secondary cluster membership per chunk

Step 3: Contrastive LLM Profiling
  profile_all_clusters()      — two-pass LLM profiling with neighbor context

Orchestrator:
  run_clustering()            — called by the Flask endpoint

LLM backend: OpenAI (gpt-4o-mini)
Ollama calls are commented out below each OpenAI equivalent.
"""

import os
import gc
import re
import json
import time
import random
import logging
import textwrap
from collections import defaultdict

import numpy as np
import torch
import joblib
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from openai import OpenAI

from app.config import (
    DEVICE,
    EMBEDDINGS_PATH, EMBEDDINGS_5D_PATH,
    GMM_MODEL_PATH, GMM_SCALER_PATH,
    CLUSTER_ASSIGN_PATH, ENRICHED_METADATA_PATH,
    CLUSTER_PROFILES_PATH, PATH_A_MANIFEST_PATH,
    CHUNKS_PATH, METADATA_PATH, INDEX_PATH,
    UMAP_N_COMPONENTS, UMAP_METRIC, UMAP_RANDOM_STATE,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    GMM_K_MIN, GMM_K_MAX, GMM_N_INIT, GMM_MAX_ITER,
    GMM_REG_COVAR, GMM_RANDOM_STATE, GMM_COVARIANCE_TYPE,
    SECONDARY_PROB_THRESHOLD,
    PROFILE_CHUNKS_PER_CLUS, PROFILE_MAX_CHUNK_CHARS,
    PROFILE_TEMPERATURE, PROFILE_MAX_RETRIES,
    OPENAI_API_KEY, LLM_MODEL_NAME,
    RANDOM_SEED, TOP_REPRESENTATIVE,
)
from app.preprocess import load_json, save_json

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
# OLLAMA_HOST = "http://localhost:11434"

# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def fit_umap_5d(embeddings: np.ndarray) -> tuple[np.ndarray, umap.UMAP]:
    """
    Reduces 1024D BGE-M3 embeddings to 5D for GMM soft clustering.
    5D preserves enough local geometry for GMM while compressing ~200×.
    """
    logger.info("Fitting 5D UMAP on %d vectors …", embeddings.shape[0])

    reducer = umap.UMAP(
        n_components = UMAP_N_COMPONENTS,
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = UMAP_RANDOM_STATE,
        low_memory   = True,
        verbose      = False,
    )

    t0      = time.perf_counter()
    emb_5d  = reducer.fit_transform(embeddings).astype("float32")
    elapsed = time.perf_counter() - t0

    logger.info("UMAP complete in %.1fs — shape %s", elapsed, emb_5d.shape)
    return emb_5d, reducer


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_embeddings(emb_5d: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler     = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_5d).astype("float32")
    joblib.dump(scaler, GMM_SCALER_PATH)
    logger.info("Scaler saved → %s", GMM_SCALER_PATH)
    return emb_scaled, scaler


# ---------------------------------------------------------------------------
# BIC sweep
# ---------------------------------------------------------------------------

def bic_sweep(emb_scaled: np.ndarray) -> tuple[list[int], list[float], list[float]]:
    ks, bics, aics = [], [], []
    logger.info("BIC sweep k=%d..%d …", GMM_K_MIN, GMM_K_MAX)

    for k in range(GMM_K_MIN, GMM_K_MAX + 1):
        gmm = GaussianMixture(
            n_components    = k,
            covariance_type = GMM_COVARIANCE_TYPE,
            n_init          = GMM_N_INIT,
            max_iter        = GMM_MAX_ITER,
            reg_covar       = GMM_REG_COVAR,
            random_state    = GMM_RANDOM_STATE,
        )
        gmm.fit(emb_scaled)
        ks.append(k)
        bics.append(gmm.bic(emb_scaled))
        aics.append(gmm.aic(emb_scaled))
        logger.debug("k=%d  BIC=%.2f  converged=%s", k, bics[-1], gmm.converged_)

    best_k = ks[int(np.argmin(bics))]
    logger.info("Best k=%d (BIC=%.2f)", best_k, min(bics))
    return ks, bics, aics


# ---------------------------------------------------------------------------
# Final GMM
# ---------------------------------------------------------------------------

def fit_final_gmm(emb_scaled: np.ndarray, k: int) -> GaussianMixture:
    logger.info("Fitting final GMM k=%d …", k)
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = GMM_COVARIANCE_TYPE,
        n_init          = GMM_N_INIT * 2,
        max_iter        = GMM_MAX_ITER,
        reg_covar       = GMM_REG_COVAR,
        random_state    = GMM_RANDOM_STATE,
    )
    gmm.fit(emb_scaled)
    logger.info("GMM converged=%s  n_iter=%d  lower_bound=%.4f",
                gmm.converged_, gmm.n_iter_, gmm.lower_bound_)
    return gmm


# ---------------------------------------------------------------------------
# Soft assignments
# ---------------------------------------------------------------------------

def compute_soft_assignments(
    gmm       : GaussianMixture,
    emb_scaled: np.ndarray,
    chunks    : list[dict],
    threshold : float = SECONDARY_PROB_THRESHOLD,
) -> tuple[np.ndarray, list[dict]]:
    logger.info("Computing soft assignments (threshold=%.2f) …", threshold)
    proba_matrix = gmm.predict_proba(emb_scaled)

    assignments = []
    n_bridge    = 0

    for i, (chunk, proba_row) in enumerate(zip(chunks, proba_matrix)):
        primary_idx  = int(np.argmax(proba_row))
        primary_prob = float(proba_row[primary_idx])

        secondary = [
            {"cluster_id": int(j), "probability": round(float(p), 6)}
            for j, p in enumerate(proba_row)
            if j != primary_idx and p > threshold
        ]
        secondary.sort(key=lambda x: x["probability"], reverse=True)

        if secondary:
            n_bridge += 1

        full_proba = {
            str(j): round(float(p), 6)
            for j, p in enumerate(proba_row)
        }

        assignments.append({
            "index"              : i,
            "chunk_id"           : chunk["chunk_id"],
            "source_doc"         : chunk["source_doc"],
            "primary_cluster"    : primary_idx,
            "primary_probability": round(primary_prob, 6),
            "secondary_clusters" : secondary,
            "is_bridge_chunk"    : len(secondary) > 0,
            "full_probabilities" : full_proba,
            "text"               : chunk["text"],
        })

    logger.info("Soft assignments done — %d chunks, %d bridge (%.1f%%)",
                len(assignments), n_bridge, n_bridge / len(assignments) * 100)
    return proba_matrix, assignments


# ---------------------------------------------------------------------------
# Enriched metadata
# ---------------------------------------------------------------------------

def build_enriched_metadata(
    chunks     : list[dict],
    assignments: list[dict],
) -> list[dict]:
    assign_by_id = {a["chunk_id"]: a for a in assignments}
    enriched = []
    missing  = 0

    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        a   = assign_by_id.get(cid)

        if a is None:
            missing += 1
            enriched.append({
                "index"               : i,
                "chunk_id"            : cid,
                "source_doc"          : chunk["source_doc"],
                "text"                : chunk["text"],
                "cluster_ids"         : [-1],
                "primary_cluster"     : -1,
                "primary_probability" : 0.0,
                "secondary_clusters"  : [],
                "is_bridge_chunk"     : False,
                "full_probabilities"  : {},
            })
            continue

        secondary_ids = [s["cluster_id"] for s in a["secondary_clusters"]]
        cluster_ids   = [a["primary_cluster"]] + secondary_ids

        enriched.append({
            "index"               : i,
            "chunk_id"            : cid,
            "source_doc"          : chunk["source_doc"],
            "text"                : chunk["text"],
            "cluster_ids"         : cluster_ids,
            "primary_cluster"     : a["primary_cluster"],
            "primary_probability" : a["primary_probability"],
            "secondary_clusters"  : a["secondary_clusters"],
            "is_bridge_chunk"     : a["is_bridge_chunk"],
            "full_probabilities"  : a["full_probabilities"],
        })

    if missing:
        logger.warning("%d chunks had no assignment — marked cluster -1", missing)

    return enriched


# ---------------------------------------------------------------------------
# Cluster centroids + nearest neighbors
# ---------------------------------------------------------------------------

def compute_cluster_centroids_5d(
    cluster_groups: dict[int, list[dict]],
    emb_5d        : np.ndarray,
) -> dict[int, np.ndarray]:
    centroids = {}
    for cid, members in cluster_groups.items():
        indices        = [r["index"] for r in members]
        vecs           = emb_5d[indices]
        centroids[cid] = vecs.mean(axis=0)
    return centroids


def compute_nearest_neighbors_per_cluster(
    centroids: dict[int, np.ndarray],
    top_n    : int = 2,
) -> dict[int, list[int]]:
    cluster_ids     = sorted(centroids.keys())
    id_to_idx       = {cid: i for i, cid in enumerate(cluster_ids)}
    centroid_matrix = np.stack([centroids[cid] for cid in cluster_ids])

    norms      = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    normed     = centroid_matrix / (norms + 1e-10)
    sim_matrix = normed @ normed.T

    neighbors = {}
    for cid in cluster_ids:
        i        = id_to_idx[cid]
        sims     = sim_matrix[i].copy()
        sims[i]  = -999.0
        top_idxs = np.argsort(sims)[::-1][:top_n]
        neighbors[cid] = [cluster_ids[idx] for idx in top_idxs]
    return neighbors


# ---------------------------------------------------------------------------
# LLM profiling helpers
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = (
    "You are an expert data taxonomist specializing in semantic "
    "analysis of text corpora. Your responses are always valid, "
    "parseable JSON with no additional commentary."
)


def build_cluster_context(
    members        : list[dict],
    cluster_id     : int,
    n_samples      : int = PROFILE_CHUNKS_PER_CLUS,
    max_chunk_chars: int = PROFILE_MAX_CHUNK_CHARS,
) -> tuple[str, list[str]]:
    rng    = random.Random(RANDOM_SEED + cluster_id)
    n_top  = min(2, len(members))
    top    = members[:n_top]
    rest   = members[n_top:]
    n_rand = min(n_samples - n_top, len(rest))
    rand   = rng.sample(rest, n_rand) if n_rand > 0 else []

    sampled     = top + rand
    sampled_ids = [r["chunk_id"] for r in sampled]

    parts = []
    for i, record in enumerate(sampled, start=1):
        text = record["text"].strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + "…"
        src  = os.path.basename(record["source_doc"])
        prob = record["primary_probability"]
        parts.append(
            f"[Chunk {i} | source: {src} | cluster_prob: {prob:.3f}]\n{text}"
        )
    return "\n\n".join(parts), sampled_ids


def build_profile_prompt(
    cluster_id       : int,
    context_str      : str,
    n_clusters       : int,
    neighbor_ids     : list[int],
    neighbor_profiles: dict,
) -> str:
    if neighbor_profiles and neighbor_ids:
        neighbor_lines = []
        for nid in neighbor_ids:
            nid_str = str(nid)
            if nid_str in neighbor_profiles:
                np_ = neighbor_profiles[nid_str]["profile"]
                neighbor_lines.append(
                    f"  Cluster {nid} (geometrically adjacent):\n"
                    f"    Theme   : {np_['primary_theme']}\n"
                    f"    Entities: {', '.join(np_['key_entities'])}\n"
                    f"    Edge    : {np_['contrastive_edge'][:120]}…"
                )
        neighbor_block = (
            "GEOMETRICALLY ADJACENT CLUSTERS\n"
            "-------------------------------------------\n"
            + "\n".join(neighbor_lines)
            + "\n\nYour contrastive_edge MUST explain what separates this cluster "
              "from the adjacent clusters above."
        ) if neighbor_lines else (
            f"NOTE: Geometrically closest cluster IDs are {neighbor_ids}. "
            f"Their profiles are not yet available. Infer contrast from text content."
        )
    else:
        neighbor_block = ""

    return textwrap.dedent(f"""
        TASK
        ----
        You are analyzing Cluster {cluster_id} out of {n_clusters} total clusters.

        TEXT CHUNKS
        -----------
        {context_str}

        {neighbor_block}

        INSTRUCTIONS
        ------------
        Return a JSON object with EXACTLY these three keys:

        {{
          "primary_theme": "<concise 5-word title>",
          "key_entities": ["<e1>", "<e2>", "<e3>", "<e4>", "<e5>"],
          "contrastive_edge": "<Two sentences explaining what makes this cluster
                               distinct from its geometrically nearest neighbors.>"
        }}

        Return ONLY the JSON object.
    """).strip()


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    repaired  = {}
    theme_m   = re.search(r'"primary_theme"\s*:\s*"([^"]+)"', raw)
    if theme_m:
        repaired["primary_theme"] = theme_m.group(1)
    entities_m = re.search(r'"key_entities"\s*:\s*\[([^\]]+)\]', raw)
    if entities_m:
        entities = re.findall(r'"([^"]+)"', entities_m.group(1))
        if entities:
            repaired["key_entities"] = entities
    edge_m = re.search(r'"contrastive_edge"\s*:\s*"([^"]+)"', raw, re.DOTALL)
    if edge_m:
        repaired["contrastive_edge"] = edge_m.group(1)
    if len(repaired) == 3:
        return repaired

    raise ValueError(f"Could not extract JSON. Raw: {raw[:300]}")


# ---------------------------------------------------------------------------
# OpenAI call for cluster profiling
# ---------------------------------------------------------------------------

def call_openai_for_profile(prompt: str) -> tuple[dict, dict]:
    """
    Calls OpenAI chat completions to generate a cluster profile.
    Returns (profile_dict, meta_dict).
    """
    client     = get_openai_client()
    last_error = None

    for attempt in range(1, PROFILE_MAX_RETRIES + 1):
        current_prompt = prompt
        if attempt > 1:
            current_prompt += (
                "\n\nIMPORTANT: Return ONLY a raw JSON object starting with { "
                "and ending with }. No markdown, no explanation."
            )

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model       = LLM_MODEL_NAME,
                temperature = PROFILE_TEMPERATURE,
                messages    = [
                    {"role": "system", "content": SYSTEM_PREAMBLE},
                    {"role": "user",   "content": current_prompt},
                ],
                response_format = {"type": "json_object"},
            )
        except Exception as e:
            last_error = str(e)
            logger.warning("OpenAI call failed (attempt %d): %s", attempt, e)
            time.sleep(2)
            continue

        elapsed  = time.perf_counter() - t0
        raw_text = response.choices[0].message.content.strip()

        try:
            profile = _extract_json(raw_text)
            for key in ("primary_theme", "key_entities", "contrastive_edge"):
                if key not in profile:
                    raise ValueError(f"Missing key: {key!r}")
            if not isinstance(profile["key_entities"], list):
                profile["key_entities"] = [str(profile["key_entities"])]

            meta = {
                "attempt"            : attempt,
                "elapsed_s"          : round(elapsed, 2),
                "prompt_tokens"      : response.usage.prompt_tokens,
                "completion_tokens"  : response.usage.completion_tokens,
            }
            return profile, meta

        except (ValueError, KeyError) as e:
            last_error = str(e)
            time.sleep(1)

    raise RuntimeError(
        f"Failed after {PROFILE_MAX_RETRIES} attempts. Last error: {last_error}"
    )

# ---------------------------------------------------------------------------
# Ollama call — commented out
# ---------------------------------------------------------------------------
# def call_ollama_for_profile(prompt: str) -> tuple[dict, dict]:
#     payload_base = {
#         "model" : LLM_MODEL_NAME,
#         "stream": False,
#         "format": "json",
#         "options": {
#             "temperature"   : PROFILE_TEMPERATURE,
#             "num_predict"   : PROFILE_NUM_PREDICT,
#             "top_p"         : 0.9,
#             "repeat_penalty": 1.1,
#         },
#     }
#     last_error = None
#     for attempt in range(1, PROFILE_MAX_RETRIES + 1):
#         ...
#         r = requests.post(f"{OLLAMA_HOST}/api/generate", json=..., timeout=PROFILE_TIMEOUT_S)
#         ...


def _normalise_entities(profile: dict) -> None:
    entities = [str(e).strip() for e in profile.get("key_entities", []) if e]
    entities = entities[:5]
    while len(entities) < 5:
        entities.append("—")
    profile["key_entities"] = entities


def _make_entry(cid, profile, meta, sampled_ids, members):
    return {
        "cluster_id"       : cid,
        "profile"          : profile,
        "sampled_chunk_ids": sampled_ids,
        "n_members"        : len(members),
        "avg_prob"         : round(
            sum(r["primary_probability"] for r in members) / len(members), 4),
        "llm_meta"         : meta,
    }


def profile_all_clusters(
    cluster_groups   : dict[int, list[dict]],
    cluster_neighbors: dict[int, list[int]],
) -> dict:
    """
    Two-pass contrastive profiling using OpenAI.
    Pass 1: all clusters profiled without neighbor context.
    Pass 2: re-profile injecting geometrically nearest neighbor profiles.
    """
    n_clusters = len(cluster_groups)
    profiles   = {}
    errors     = {}
    total_time = 0.0

    def _profile_one(cid, existing_profiles):
        members          = cluster_groups[cid]
        ctx, sampled_ids = build_cluster_context(members, cid)
        neighbor_ids     = cluster_neighbors.get(cid, [])
        prompt           = build_profile_prompt(
            cid, ctx, n_clusters, neighbor_ids, existing_profiles
        )
        # OpenAI
        return call_openai_for_profile(prompt), sampled_ids, members
        # Ollama (commented out):
        # return call_ollama_for_profile(prompt), sampled_ids, members

    logger.info("PASS 1 — profiling %d clusters with OpenAI…", n_clusters)
    for cid in sorted(cluster_groups.keys()):
        try:
            (profile, meta), sampled_ids, members = _profile_one(cid, {})
            _normalise_entities(profile)
            profiles[str(cid)] = _make_entry(cid, profile, meta, sampled_ids, members)
            total_time += meta["elapsed_s"]
            logger.info("Cluster %d profiled in %.2fs", cid, meta["elapsed_s"])
        except RuntimeError as e:
            errors[str(cid)] = {"cluster_id": cid, "error": str(e)}
            logger.error("Cluster %d profiling failed: %s", cid, e)

    logger.info("PASS 2 — re-profiling with neighbor context…")
    for cid in sorted(cluster_groups.keys()):
        if str(cid) in errors:
            continue
        try:
            (profile, meta), sampled_ids, members = _profile_one(cid, profiles)
            _normalise_entities(profile)
            profiles[str(cid)] = _make_entry(cid, profile, meta, sampled_ids, members)
            total_time += meta["elapsed_s"]
            logger.info("Cluster %d re-profiled in %.2fs", cid, meta["elapsed_s"])
        except RuntimeError as e:
            errors[str(cid)] = {"cluster_id": cid, "error": str(e)}
            logger.error("Cluster %d pass-2 profiling failed: %s", cid, e)

    logger.info("Profiling done — %d profiles, %d errors, %.1fs total",
                len(profiles), len(errors), total_time)

    return {
        "profiles"    : profiles,
        "errors"      : errors,
        "n_clusters"  : n_clusters,
        "model"       : LLM_MODEL_NAME,
        "timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
    }


# ---------------------------------------------------------------------------
# Orchestrator — called by Flask endpoint
# ---------------------------------------------------------------------------

def clustering_artefacts_exist() -> dict[str, bool]:
    return {
        "embeddings_5d": os.path.exists(EMBEDDINGS_5D_PATH),
        "gmm_model"    : os.path.exists(GMM_MODEL_PATH),
        "gmm_scaler"   : os.path.exists(GMM_SCALER_PATH),
        "assignments"  : os.path.exists(CLUSTER_ASSIGN_PATH),
        "enriched"     : os.path.exists(ENRICHED_METADATA_PATH),
        "profiles"     : os.path.exists(CLUSTER_PROFILES_PATH),
    }


def run_clustering(force_rerun: bool = False) -> dict:
    """
    Full offline clustering + profiling pipeline.
    Requires preprocessing artefacts to already exist.
    """
    existing = clustering_artefacts_exist()
    if all(existing.values()) and not force_rerun:
        logger.info("All clustering artefacts exist. Skipping.")
        with open(CLUSTER_PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        return {
            "status"    : "already_done",
            "message"   : "Clustering artefacts present. Pass force_rerun=true to re-run.",
            "n_clusters": profiles.get("n_clusters", 0),
        }

    if not os.path.exists(EMBEDDINGS_PATH):
        return {
            "status" : "error",
            "message": "embeddings.npy not found. Run preprocessing first.",
        }

    t_total = time.perf_counter()

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    chunks     = load_json(CHUNKS_PATH)

    # Step 1: UMAP
    emb_5d, _ = fit_umap_5d(embeddings)
    np.save(EMBEDDINGS_5D_PATH, emb_5d)

    # Step 2: Scale
    emb_scaled, _ = scale_embeddings(emb_5d)

    # Step 3: BIC sweep
    ks, bics, _ = bic_sweep(emb_scaled)
    best_k      = ks[int(np.argmin(bics))]

    # Step 4: Final GMM
    final_gmm = fit_final_gmm(emb_scaled, best_k)
    joblib.dump(final_gmm, GMM_MODEL_PATH)

    # Step 5: Soft assignments
    _, assignments = compute_soft_assignments(final_gmm, emb_scaled, chunks)
    save_json(assignments, CLUSTER_ASSIGN_PATH)

    # Step 6: Enriched metadata
    enriched = build_enriched_metadata(chunks, assignments)
    save_json(enriched, ENRICHED_METADATA_PATH)

    # Step 7: Cluster centroids + neighbors
    cluster_groups: dict[int, list[dict]] = defaultdict(list)
    for record in enriched:
        cid = record["primary_cluster"]
        if cid != -1:
            cluster_groups[cid].append(record)
    for cid in cluster_groups:
        cluster_groups[cid].sort(
            key=lambda r: r["primary_probability"], reverse=True
        )

    centroids = compute_cluster_centroids_5d(dict(cluster_groups), emb_5d)
    neighbors = compute_nearest_neighbors_per_cluster(centroids, top_n=2)

    # Step 8: LLM profiling (OpenAI)
    profiling_results = profile_all_clusters(dict(cluster_groups), neighbors)
    save_json(profiling_results, CLUSTER_PROFILES_PATH)

    del embeddings, emb_5d, emb_scaled
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_total
    logger.info("Clustering complete in %.1fs", elapsed)

    return {
        "status"    : "done",
        "message"   : f"Clustering complete in {elapsed:.1f}s",
        "n_clusters": best_k,
        "elapsed_s" : round(elapsed, 2),
        "errors"    : profiling_results.get("errors", {}),
    }