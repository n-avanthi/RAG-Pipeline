"""
vanilla_rag_benchmark.py
------------------------
Standalone vanilla RAG evaluation script.

Run AFTER benchmark.py (which produces benchmarks.json):

    python benchmark.py          # AC-RAG — runs first
    python vanilla_rag_benchmark.py   # Vanilla RAG — loads same queries

Outputs:
  - vanilla_benchmarks.json    (vanilla results)
  - Full comparison table printed to terminal

Design
------
Vanilla RAG is the strongest possible baseline:
  - Same BGE-M3 embeddings, same corpus, same chunk size as AC-RAG
  - IndexFlatIP (brute-force exact search) — no recall approximation error.
    This is deliberately the strongest opponent: vanilla retrieves the
    globally best K chunks by pure similarity.
  - NO cluster routing, NO MMR, NO diversity enforcement.
  - Same total chunks retrieved as AC-RAG (AC_N_CLUSTERS * AC_TOP_K = 6).
  - Same OpenAI model for answer generation.
  - Identical queries, nuggets, and metrics as benchmark.py.

Ollama answer generation is commented out but preserved for reference.

Research note: Using IndexFlatIP (exact) rather than HNSW for vanilla
makes the comparison maximally defensible — vanilla has no recall penalty.
Any diversity advantage AC-RAG shows is purely from its routing strategy,
not from HNSW approximation error in the baseline.
"""

import os
import sys
import json
import time
import logging
import warnings
from collections import defaultdict

import numpy as np
import torch
import faiss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse AC-RAG's infrastructure (same embeddings, same encoder)
from app.config import (
    EMBEDDINGS_PATH,
    INDEX_PATH,
    ENRICHED_METADATA_PATH,
    CLUSTER_PROFILES_PATH,
    OPENAI_API_KEY,
    LLM_MODEL_NAME,
)
from app.preprocess import get_bge_model, load_json
import app.rag_pipeline as _rag

from app.rag_pipeline import (
    load_pipeline_state,
    is_pipeline_ready,
)

# Always read through the module so we get the live value after
# load_pipeline_state() has populated these globals
def _enriched():          return _rag._enriched
def _embeddings_matrix(): return _rag._embeddings_matrix
def _index_to_record():   return _rag._index_to_record
def _valid_cluster_ids(): return _rag._valid_cluster_ids

# ── OpenAI client ─────────────────────────────────────────────────────────────
from openai import OpenAI

_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── Ollama client — commented out ─────────────────────────────────────────────
# import requests
# OLLAMA_HOST    = "http://localhost:11434"
# LLM_MODEL_NAME_OLLAMA = "mistral:latest"


# =============================================================================
# BENCHMARK PARAMETERS — must mirror benchmark.py exactly
# =============================================================================

AC_TOP_K      = 3       # chunks per cluster in AC-RAG
AC_N_CLUSTERS = 2       # clusters AC-RAG selects
VANILLA_TOP_K = AC_TOP_K * AC_N_CLUSTERS   # = 6  (size-matched)
LAMBDA        = 0.7     # used in AC-RAG MMR (not used in vanilla, recorded for reference)

# Output paths
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
VANILLA_BENCH_PATH  = os.path.join(SCRIPT_DIR, "vanilla_benchmarks.json")
ACRAG_BENCH_PATH    = os.path.join(SCRIPT_DIR, "benchmarks.json")


# =============================================================================
# BENCHMARK QUERIES — identical to benchmark.py
# =============================================================================

BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": (
            "How do PPO's clipping objective and Adam's adaptive learning rates "
            "together stabilise training, and what failure modes does each introduce?"
        ),
        "nuggets": [
            "clipping", "policy", "surrogate", "epsilon",
            "adaptive", "momentum", "learning rate", "gradient",
        ],
        "spans_clusters": ["Reinforcement Learning", "Optimisation"],
    },
    {
        "id": "Q2",
        "query": (
            "What do RAG's external retrieval and transformer self-attention have "
            "in common architecturally, and where do they fundamentally diverge "
            "in how they surface relevant information?"
        ),
        "nuggets": [
            "retrieval", "document", "dense", "index",
            "attention", "key", "value", "query",
        ],
        "spans_clusters": ["RAG / Retrieval", "Transformers"],
    },
    {
        "id": "Q3",
        "query": (
            "Compare the memory complexity trade-offs in Longformer, Linformer, "
            "and FlashAttention against the quadratic cost of standard attention "
            "as sequence length grows."
        ),
        "nuggets": [
            "linear", "sparse", "sliding window", "tiling",
            "quadratic", "sequence length", "softmax", "matrix",
        ],
        "spans_clusters": ["Efficient Attention", "Transformers"],
    },
    {
        "id": "Q4",
        "query": (
            "How does CLIP's image-text contrastive objective differ from BERT's "
            "masked language modelling in what the learned representations capture?"
        ),
        "nuggets": [
            "contrastive", "image", "text", "alignment",
            "masked", "token", "prediction", "bidirectional",
        ],
        "spans_clusters": ["Vision / Multimodal", "Language Models"],
    },
    {
        "id": "Q5",
        "query": (
            "InstructGPT applies RLHF using PPO — what properties of PPO make it "
            "suitable for language model fine-tuning, and what alignment risks does "
            "this training procedure introduce?"
        ),
        "nuggets": [
            "reward model", "human feedback", "KL divergence", "alignment",
            "clipping", "policy gradient", "on-policy", "instability",
        ],
        "spans_clusters": ["Alignment / RLHF", "Reinforcement Learning"],
    },
    {
        "id": "Q6",
        "query": (
            "Why is pre-norm layer normalisation preferred over post-norm in deep "
            "transformers, and how does this choice interact with residual "
            "connections and gradient flow during training?"
        ),
        "nuggets": [
            "gradient", "stability", "vanishing", "depth",
            "residual", "layer norm", "pre-norm", "post-norm",
        ],
        "spans_clusters": ["Optimisation", "Transformer Architecture"],
    },
    {
        "id": "Q7",
        "query": (
            "How do DINO's self-supervised vision pretraining and DreamerV2's "
            "world-model learning both exploit latent representations, and what "
            "makes their objectives structurally different?"
        ),
        "nuggets": [
            "self-supervised", "patch", "teacher", "distillation",
            "world model", "latent", "imagination", "reconstruction",
        ],
        "spans_clusters": ["Vision / Multimodal", "Reinforcement Learning"],
    },
]


# =============================================================================
# VANILLA RETRIEVAL — IndexFlatIP (exact brute-force, no approximation)
# =============================================================================

_flat_index: faiss.IndexFlatIP | None = None


def _get_flat_index() -> faiss.IndexFlatIP:
    """
    Builds or returns a cached IndexFlatIP over the full embeddings matrix.

    Why FlatIP and not HNSW?
    ------------------------
    IndexFlatIP is exact — it finds the globally best K chunks with
    zero recall error. This is the strongest possible vanilla baseline:
    if AC-RAG wins against exact vanilla retrieval, the win is
    unambiguously attributable to its routing strategy, not to any
    recall deficit in the baseline.
    """
    global _flat_index
    if _flat_index is None:
        print("  Building IndexFlatIP (exact search)...", end=" ", flush=True)
        emb = _embeddings_matrix().astype("float32")
        _flat_index = faiss.IndexFlatIP(emb.shape[1])
        _flat_index.add(emb)
        print(f"done. {_flat_index.ntotal} vectors indexed.")
    return _flat_index


def vanilla_retrieve(q_vec: np.ndarray, top_k: int = VANILLA_TOP_K) -> dict:
    """
    Flat retrieval: IndexFlatIP global search → top-k, no cluster filter.

    Returns dict[cluster_id -> list[chunk]] in the same format as AC-RAG
    so identical metric functions run on both without modification.
    Cluster IDs are the primary_cluster assignments from the enriched metadata.
    """
    idx = _get_flat_index()
    q   = q_vec.reshape(1, -1).astype("float32")
    distances, indices = idx.search(q, top_k)

    grouped: dict[int, list[dict]] = defaultdict(list)
    idx_to_rec = _index_to_record()
    for score, fidx in zip(distances[0], indices[0]):
        if fidx == -1:
            continue
        record = dict(idx_to_rec[fidx])
        record["similarity_score"] = round(float(score), 6)
        grouped[record["primary_cluster"]].append(record)

    return dict(grouped)


# =============================================================================
# EMBED QUERY — reuses AC-RAG's BGE-M3 model
# =============================================================================

def embed_query(query: str) -> np.ndarray:
    model = get_bge_model()
    with torch.no_grad():
        vec = model.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
    return vec[0].astype("float32")


# =============================================================================
# METRICS — identical implementations to benchmark.py
# =============================================================================

def _all_indices(retrieved: dict) -> list[int]:
    return [c["index"] for chunks in retrieved.values() for c in chunks]


def _all_text(retrieved: dict) -> str:
    return " ".join(
        c["text"].lower() for chunks in retrieved.values() for c in chunks
    )


def compute_mrr(retrieved: dict, q_vec: np.ndarray, k: int = 8) -> float:
    """MRR@k. Pseudo-relevance: cosine sim >= 0.4 → relevant."""
    indices = _all_indices(retrieved)
    if not indices:
        return 0.0
    emb    = _embeddings_matrix()
    vecs   = emb[indices]
    sims   = vecs @ q_vec
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    for rank, (_, sim) in enumerate(ranked[:k], start=1):
        if float(sim) >= 0.4:
            return round(1.0 / rank, 4)
    return 0.0


def compute_relevance(retrieved: dict, q_vec: np.ndarray) -> float:
    """Mean cosine similarity between query and all retrieved chunks."""
    indices = _all_indices(retrieved)
    if not indices:
        return 0.0
    emb  = _embeddings_matrix()
    vecs = emb[indices]
    return round(float(np.mean(vecs @ q_vec)), 4)


def compute_diversity(retrieved: dict) -> float:
    """Mean pairwise cosine distance. Range [0,2]. Higher = more diverse."""
    indices = _all_indices(retrieved)
    n = len(indices)
    if n < 2:
        return 0.0
    emb     = _embeddings_matrix()
    vecs    = emb[indices]
    sim_mat = vecs @ vecs.T
    pairs   = [1.0 - float(sim_mat[i, j]) for i in range(n) for j in range(i+1, n)]
    return round(float(np.mean(pairs)), 4)


def compute_redundancy(retrieved: dict) -> float:
    """Mean pairwise cosine similarity. Lower = less redundant."""
    indices = _all_indices(retrieved)
    n = len(indices)
    if n < 2:
        return 1.0
    emb     = _embeddings_matrix()
    vecs    = emb[indices]
    sim_mat = vecs @ vecs.T
    pairs   = [float(sim_mat[i, j]) for i in range(n) for j in range(i+1, n)]
    return round(float(np.mean(pairs)), 4)


def compute_source_concentration(retrieved: dict) -> float:
    """Fraction of chunks from the most dominant source. Lower = better."""
    counts: dict[str, int] = {}
    total = 0
    for chunks in retrieved.values():
        for c in chunks:
            src = os.path.basename(c["source_doc"])
            counts[src] = counts.get(src, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    return round(max(counts.values()) / total, 4)


def compute_cohesion(retrieved: dict) -> float:
    """Mean intra-cluster cosine similarity. Reported only, not competitive."""
    cohesions = []
    emb = _embeddings_matrix()
    for cid, chunks in retrieved.items():
        if len(chunks) < 2:
            cohesions.append(1.0)
            continue
        idxs    = [c["index"] for c in chunks]
        vecs    = emb[idxs]
        sim_mat = vecs @ vecs.T
        n       = len(idxs)
        pairs   = [float(sim_mat[i, j]) for i in range(n) for j in range(i+1, n)]
        cohesions.append(float(np.mean(pairs)))
    return round(float(np.mean(cohesions)), 4) if cohesions else 0.0


def compute_nugget_hit_rate(retrieved: dict, nuggets: list[str]) -> float:
    """Fraction of nuggets found in retrieved text. Higher = better coverage."""
    if not nuggets:
        return 0.0
    text = _all_text(retrieved)
    hits = sum(1 for n in nuggets if n.lower() in text)
    return round(hits / len(nuggets), 4)


def compute_rd_score(relevance: float, diversity: float) -> float:
    """Harmonic mean of relevance and normalised diversity. Headline metric."""
    norm_div = diversity / 2.0
    if relevance + norm_div == 0:
        return 0.0
    return round(2 * (relevance * norm_div) / (relevance + norm_div), 4)


METRIC_KEYS = [
    "mrr", "relevance", "diversity", "redundancy",
    "source_concentration", "cohesion", "nugget_hit_rate", "rd_score",
]

COMPETITIVE_METRICS = [
    "mrr", "relevance", "diversity", "redundancy",
    "source_concentration", "nugget_hit_rate", "rd_score",
]

METRIC_LABELS = {
    "mrr":                  "MRR@k",
    "relevance":            "Relevance",
    "diversity":            "Diversity",
    "redundancy":           "Redundancy",
    "source_concentration": "Source Conc.",
    "cohesion":             "Cohesion *",
    "nugget_hit_rate":      "Nugget Hit",
    "rd_score":             "R-D Score",
}

HIGHER_IS_BETTER = {
    "mrr":                  True,
    "relevance":            True,
    "diversity":            True,
    "redundancy":           False,
    "source_concentration": False,
    "cohesion":             True,
    "nugget_hit_rate":      True,
    "rd_score":             True,
}


def run_metrics(retrieved: dict, q_vec: np.ndarray, nuggets: list[str]) -> dict:
    rel = compute_relevance(retrieved, q_vec)
    div = compute_diversity(retrieved)
    return {
        "mrr":                  compute_mrr(retrieved, q_vec),
        "relevance":            rel,
        "diversity":            div,
        "redundancy":           compute_redundancy(retrieved),
        "source_concentration": compute_source_concentration(retrieved),
        "cohesion":             compute_cohesion(retrieved),
        "nugget_hit_rate":      compute_nugget_hit_rate(retrieved, nuggets),
        "rd_score":             compute_rd_score(rel, div),
        "n_chunks":             sum(len(v) for v in retrieved.values()),
        "n_clusters":           len(retrieved),
    }


# =============================================================================
# ANSWER GENERATION — OpenAI (same model as AC-RAG)
# =============================================================================

def assemble_vanilla_context(retrieved: dict, max_chars: int = 3000) -> str:
    """
    Assembles retrieved chunks into a context string.
    Format mirrors AC-RAG's assemble_rag_context() for identical LLM conditions.
    """
    lines = ["RETRIEVED CONTEXT", "=" * 55]
    total = 0
    rank  = 1
    for chunks in retrieved.values():
        for chunk in chunks:
            src    = os.path.basename(chunk["source_doc"])
            header = (f"\n[Rank {rank} | sim={chunk['similarity_score']:.4f} "
                      f"| {src} | chunk: {chunk['chunk_id']}]")
            body   = chunk["text"].strip()
            block  = f"{header}\n{body}"
            if total + len(block) > max_chars:
                lines.append(f"\n… [context truncated at {max_chars} chars]")
                return "\n".join(lines)
            lines.append(block)
            total += len(block)
            rank  += 1
    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


def generate_answer_openai(query: str, context: str) -> tuple[str, float]:
    """OpenAI answer generation — same model and prompt as AC-RAG."""
    client = get_openai_client()
    t0     = time.perf_counter()
    resp   = client.chat.completions.create(
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
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
            },
        ],
        max_tokens = 400,
    )
    elapsed = time.perf_counter() - t0
    return resp.choices[0].message.content.strip(), round(elapsed, 2)


# Ollama answer generation — commented out
# def generate_answer_ollama(query: str, context: str) -> tuple[str, float]:
#     import requests
#     prompt = (
#         "You are a knowledgeable analyst. Answer using ONLY the provided context.\n\n"
#         f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
#     )
#     t0 = time.perf_counter()
#     r  = requests.post(
#         f"{OLLAMA_HOST}/api/generate",
#         json={"model": LLM_MODEL_NAME_OLLAMA, "prompt": prompt,
#               "stream": False, "options": {"temperature": 0.3, "num_predict": 300}},
#         timeout=120,
#     )
#     r.raise_for_status()
#     elapsed = time.perf_counter() - t0
#     return r.json().get("response", "").strip(), round(elapsed, 2)


# =============================================================================
# MAIN BENCHMARK RUN
# =============================================================================

def run_vanilla_benchmark() -> dict:
    print("\n" + "=" * 72)
    print("  Vanilla RAG — Benchmark Evaluation")
    print("=" * 72)

    print("\n> Loading pipeline state (shared with AC-RAG)...", end=" ", flush=True)
    ok = load_pipeline_state()
    if not ok or not is_pipeline_ready():
        print("FAILED.\n  Run the AC-RAG processing pipeline first.")
        sys.exit(1)
    print("OK")

    print(f"  Chunks   : {len(_enriched())}")
    print(f"  Queries  : {len(BENCHMARK_QUERIES)}")
    print(f"  Vanilla  : flat top-{VANILLA_TOP_K} IndexFlatIP (exact, size-matched to AC-RAG)\n")

    results_per_query = []

    for item in BENCHMARK_QUERIES:
        qid     = item["id"]
        query   = item["query"]
        nuggets = item["nuggets"]

        print(f"  [{qid}] {query[:68]}{'...' if len(query) > 68 else ''}")

        q_vec = embed_query(query)

        t0          = time.perf_counter()
        vr_result   = vanilla_retrieve(q_vec, top_k=VANILLA_TOP_K)
        vr_latency  = round(time.perf_counter() - t0, 3)
        vr_metrics  = run_metrics(vr_result, q_vec, nuggets)
        vr_metrics["latency_s"] = vr_latency

        # Source distribution for inspection
        src_dist: dict[str, int] = {}
        for chunks in vr_result.values():
            for c in chunks:
                src = os.path.basename(c["source_doc"])
                src_dist[src] = src_dist.get(src, 0) + 1

        print(f"        R-D={vr_metrics['rd_score']:.3f}  "
              f"div={vr_metrics['diversity']:.3f}  "
              f"rel={vr_metrics['relevance']:.3f}  "
              f"nugget={vr_metrics['nugget_hit_rate']:.3f}  "
              f"srcs={len(src_dist)}")

        results_per_query.append({
            "query_id":          qid,
            "query":             query,
            "nuggets":           nuggets,
            "vanilla_rag":       vr_metrics,
            "source_dist":       src_dist,
        })

    # Aggregates
    def mean_across(key):
        return round(float(np.mean([r["vanilla_rag"][key] for r in results_per_query])), 4)

    aggregate = {
        "vanilla_rag": {m: mean_across(m) for m in METRIC_KEYS},
    }
    aggregate["vanilla_rag"]["latency_s"] = round(
        float(np.mean([r["vanilla_rag"]["latency_s"] for r in results_per_query])), 3
    )

    _print_vanilla_table(results_per_query, aggregate)

    return {
        "meta": {
            "timestamp"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_queries"    : len(BENCHMARK_QUERIES),
            "vanilla_top_k": VANILLA_TOP_K,
            "index_type"   : "IndexFlatIP (exact)",
            "model"        : LLM_MODEL_NAME,
            "note": (
                "Vanilla RAG uses exact FlatIP retrieval — no recall approximation. "
                "Queries identical to benchmark.py. Metrics identical to AC-RAG benchmark."
            ),
        },
        "aggregate":         aggregate,
        "per_query_results": results_per_query,
    }


def _print_vanilla_table(results: list, aggregate: dict):
    print("\n" + "-" * 72)
    print("  Vanilla RAG — Per-query Results")
    print("-" * 72)
    print(f"  {'Query':<8} {'R-D':>8} {'Rel':>8} {'Div':>8} {'Nugget':>8} {'SrcConc':>8}")
    print("  " + "-" * 50)
    for r in results:
        m = r["vanilla_rag"]
        print(f"  {r['query_id']:<8} {m['rd_score']:>8.4f} {m['relevance']:>8.4f} "
              f"{m['diversity']:>8.4f} {m['nugget_hit_rate']:>8.4f} "
              f"{m['source_concentration']:>8.4f}")
    print("  " + "-" * 50)
    ag = aggregate["vanilla_rag"]
    print(f"  {'MEAN':<8} {ag['rd_score']:>8.4f} {ag['relevance']:>8.4f} "
          f"{ag['diversity']:>8.4f} {ag['nugget_hit_rate']:>8.4f} "
          f"{ag['source_concentration']:>8.4f}")


# =============================================================================
# COMPARISON TABLE — loads AC-RAG results from benchmarks.json
# =============================================================================

def print_comparison(vanilla_results: dict):
    if not os.path.exists(ACRAG_BENCH_PATH):
        print(f"\n  [!] benchmarks.json not found at {ACRAG_BENCH_PATH}")
        print("      Run benchmark.py first, then re-run this script.")
        return

    with open(ACRAG_BENCH_PATH, "r", encoding="utf-8") as f:
        acrag_data = json.load(f)

    ac_agg  = acrag_data["aggregate"]["ac_rag"]
    van_agg = vanilla_results["aggregate"]["vanilla_rag"]

    print("\n" + "=" * 78)
    print("  AC-RAG  vs  Vanilla RAG  —  Head-to-Head Comparison")
    print(f"  {len(BENCHMARK_QUERIES)} queries | "
          f"AC-RAG: {AC_N_CLUSTERS} clusters × top-{AC_TOP_K} = {VANILLA_TOP_K} chunks | "
          f"Vanilla: flat top-{VANILLA_TOP_K} (IndexFlatIP)")
    print("=" * 78)

    print(f"\n  {'Metric':<20} {'AC-RAG':>10} {'Vanilla':>10} {'Δ':>10}  "
          f"{'Dir':>4}  {'Winner'}")
    print("  " + "-" * 62)

    comp_wins = 0
    for m in METRIC_KEYS:
        label   = METRIC_LABELS[m]
        ac_val  = ac_agg.get(m, 0.0)
        van_val = van_agg.get(m, 0.0)
        delta   = round(ac_val - van_val, 4)
        hib     = HIGHER_IS_BETTER[m]
        dir_s   = "↑" if hib else "↓"

        if m in COMPETITIVE_METRICS:
            ac_wins = (delta > 0) == hib
            win_s   = "AC-RAG  ✓" if ac_wins else "Vanilla ✗"
            if ac_wins:
                comp_wins += 1
        else:
            win_s = "(not competitive)"

        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {ac_val:>10.4f} {van_val:>10.4f} "
              f"{sign}{delta:>9.4f}  {dir_s:>4}  {win_s}")

    # Latency
    print("\n  " + "-" * 62)
    ac_lat  = acrag_data["aggregate"]["ac_rag"].get("latency_s", "?")
    van_lat = vanilla_results["aggregate"]["vanilla_rag"]["latency_s"]
    print(f"  {'Latency (s)':<20} {ac_lat:>10}  {van_lat:>10.3f}  "
          f"{'':>10}  {'':>4}  (AC-RAG includes router LLM call)")

    print("\n" + "=" * 78)
    print(f"  AC-RAG wins {comp_wins}/{len(COMPETITIVE_METRICS)} competitive metrics.")
    print(f"  (* Cohesion = internal clustering quality, not competitive.)")
    print(f"  Δ = AC-RAG − Vanilla. Positive = AC-RAG higher.")
    print("=" * 78)

    # Per-query R-D comparison
    if "per_query_results" in acrag_data:
        ac_per_q  = {r["query_id"]: r["ac_rag"]["rd_score"]
                     for r in acrag_data["per_query_results"]}
        van_per_q = {r["query_id"]: r["vanilla_rag"]["rd_score"]
                     for r in vanilla_results["per_query_results"]}

        print("\n  Per-query R-D Score Breakdown:")
        print(f"  {'Query':<8} {'AC-RAG':>10} {'Vanilla':>10} {'Δ':>10}  {'Winner'}")
        print("  " + "-" * 50)
        for item in BENCHMARK_QUERIES:
            qid    = item["id"]
            ac_rd  = ac_per_q.get(qid, 0.0)
            van_rd = van_per_q.get(qid, 0.0)
            d      = round(ac_rd - van_rd, 4)
            win    = "AC-RAG ✓" if d > 0 else ("Vanilla ✗" if d < 0 else "Tie")
            sign   = "+" if d >= 0 else ""
            print(f"  {qid:<8} {ac_rd:>10.4f} {van_rd:>10.4f} {sign}{d:>9.4f}  {win}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    output = run_vanilla_benchmark()

    # Save vanilla results
    with open(VANILLA_BENCH_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Vanilla results saved → {VANILLA_BENCH_PATH}")

    # Print full comparison
    print_comparison(output)
    print()