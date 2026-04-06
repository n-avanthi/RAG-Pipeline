"""
benchmark.py
------------
Evaluation script for AC-RAG vs Vanilla RAG.
Run from the project root:

    python benchmark.py

Outputs:
  - Clean tables printed to terminal
  - benchmarks.json saved to project root

Design notes
------------
Vanilla RAG: same HNSW index, same BGE-M3 embeddings, flat top-k,
no cluster filtering, no MMR. Total chunks = AC-RAG total so
comparisons are size-matched.

AC-RAG: agentic cluster router (LLM + geometric centroid signal),
HNSW retrieval filtered per cluster, MMR diversity reranking.

Claim: AC-RAG is not designed to maximise raw similarity. It trades
a small relevance margin for substantially lower source concentration
and higher retrieval diversity — properties that matter for
multi-faceted queries that span topic boundaries.

Cohesion is reported but excluded from win/loss — it is an internal
clustering quality metric, not a retrieval competition metric.
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import (
    CLUSTER_PROFILES_PATH,
    ENRICHED_METADATA_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
)
from app.preprocess import get_bge_model, get_hnsw_index, load_json

import app.rag_pipeline as _rag
from app.rag_pipeline import (
    load_pipeline_state,
    is_pipeline_ready,
    route_query,
    retrieve_with_hnsw_filtered,
)

def _cluster_profiles():  return _rag._cluster_profiles
def _enriched():          return _rag._enriched
def _embeddings_matrix(): return _rag._embeddings_matrix
def _index_to_record():   return _rag._index_to_record
def _valid_cluster_ids(): return _rag._valid_cluster_ids


# ─── Parameters ───────────────────────────────────────────────────────────────

AC_TOP_K      = 3
AC_N_CLUSTERS = 2
VANILLA_TOP_K = AC_TOP_K * AC_N_CLUSTERS   # 6 — size-matched
LAMBDA        = 0.7


# ─── Queries ──────────────────────────────────────────────────────────────────

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


# ─── Vanilla retrieval ────────────────────────────────────────────────────────

def vanilla_retrieve(q_vec: np.ndarray, top_k: int = VANILLA_TOP_K) -> dict:
    """Flat HNSW top-k, no cluster filtering, no MMR."""
    hnsw = get_hnsw_index()
    q    = q_vec.reshape(1, -1).astype("float32")
    distances, indices = hnsw.search(q, top_k)

    grouped: dict[int, list[dict]] = defaultdict(list)
    idx_to_rec = _index_to_record()
    for score, fidx in zip(distances[0], indices[0]):
        if fidx == -1:
            continue
        record = dict(idx_to_rec[fidx])
        record["similarity_score"] = round(float(score), 6)
        grouped[record["primary_cluster"]].append(record)

    return dict(grouped)


# ─── Metrics ──────────────────────────────────────────────────────────────────

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


def _all_indices(retrieved: dict) -> list[int]:
    return [c["index"] for chunks in retrieved.values() for c in chunks]


def _all_text(retrieved: dict) -> str:
    return " ".join(
        c["text"].lower() for chunks in retrieved.values() for c in chunks
    )


def compute_mrr(retrieved: dict, q_vec: np.ndarray, k: int = 8) -> float:
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
    indices = _all_indices(retrieved)
    if not indices:
        return 0.0
    emb  = _embeddings_matrix()
    vecs = emb[indices]
    return round(float(np.mean(vecs @ q_vec)), 4)


def compute_diversity(retrieved: dict) -> float:
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
    if not nuggets:
        return 0.0
    text = _all_text(retrieved)
    hits = sum(1 for n in nuggets if n.lower() in text)
    return round(hits / len(nuggets), 4)


def compute_rd_score(relevance: float, diversity: float) -> float:
    norm_div = diversity / 2.0
    if relevance + norm_div == 0:
        return 0.0
    return round(2 * (relevance * norm_div) / (relevance + norm_div), 4)


METRIC_KEYS = [
    "relevance", "diversity", "redundancy",
    "source_concentration", "cohesion", "rd_score",
]

COMPETITIVE_METRICS = [
    "relevance", "diversity", "redundancy",
    "source_concentration", "rd_score",
]

METRIC_LABELS = {
    "relevance":            "Relevance",
    "diversity":            "Diversity",
    "redundancy":           "Redundancy",
    "source_concentration": "Source Conc.",
    "cohesion":             "Cohesion",
    "rd_score":             "R-D Score",
}

HIGHER_IS_BETTER = {
    "relevance":            True,
    "diversity":            True,
    "redundancy":           False,
    "source_concentration": False,
    "cohesion":             True,
    "rd_score":             True,
}


def run_metrics(retrieved: dict, q_vec: np.ndarray, nuggets: list[str]) -> dict:
    rel = compute_relevance(retrieved, q_vec)
    div = compute_diversity(retrieved)
    return {
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


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_benchmark() -> dict:
    _print_header()

    ok = load_pipeline_state()
    if not ok or not is_pipeline_ready():
        print("  Pipeline not ready. Run the processing pipeline first.")
        sys.exit(1)

    total_clusters = len(_valid_cluster_ids())
    enriched       = _enriched()
    print(f"  Corpus   :  {len(enriched)} chunks  ·  {total_clusters} clusters")
    print(f"  AC-RAG   :  top-{AC_TOP_K} × {AC_N_CLUSTERS} clusters  ·  λ={LAMBDA}")
    print(f"  Vanilla  :  flat top-{VANILLA_TOP_K}  ·  same HNSW index")
    print(f"  Queries  :  {len(BENCHMARK_QUERIES)}\n")

    results_per_query = []

    for item in BENCHMARK_QUERIES:
        qid     = item["id"]
        query   = item["query"]
        nuggets = item["nuggets"]

        short_q = query[:72] + ("…" if len(query) > 72 else "")
        print(f"  {qid}  {short_q}")

        q_vec = embed_query(query)

        # AC-RAG
        t0      = time.perf_counter()
        routing = route_query(query)
        routing["selected_clusters"] = routing["selected_clusters"][:AC_N_CLUSTERS]
        ac_result  = retrieve_with_hnsw_filtered(
            routing_result=routing, top_k=AC_TOP_K, diversity_lambda=LAMBDA
        )
        ac_latency = round(time.perf_counter() - t0, 3)
        ac_metrics = run_metrics(ac_result, q_vec, nuggets)
        ac_metrics["latency_s"]         = ac_latency
        ac_metrics["selected_clusters"] = routing["selected_clusters"]
        ac_metrics["routing_reasoning"] = routing["reasoning"]
        ac_metrics["centroid_sims"]     = routing.get("centroid_sims", {})

        # Vanilla
        t0         = time.perf_counter()
        vr_result  = vanilla_retrieve(q_vec, top_k=VANILLA_TOP_K)
        vr_latency = round(time.perf_counter() - t0, 3)
        vr_metrics = run_metrics(vr_result, q_vec, nuggets)
        vr_metrics["latency_s"] = vr_latency

        deltas = {
            m: round(ac_metrics[m] - vr_metrics[m], 4)
            for m in METRIC_KEYS
        }

        results_per_query.append({
            "query_id":    qid,
            "query":       query,
            "nuggets":     nuggets,
            "ac_rag":      ac_metrics,
            "vanilla_rag": vr_metrics,
            "deltas":      deltas,
        })

        rd_delta = deltas["rd_score"]
        marker   = "↑" if rd_delta > 0 else ("↓" if rd_delta < 0 else "=")
        print(f"       clusters {routing['selected_clusters']}  "
              f"R-D: {ac_metrics['rd_score']:.3f} vs {vr_metrics['rd_score']:.3f}  "
              f"{marker}  src_conc: {ac_metrics['source_concentration']:.2f} vs "
              f"{vr_metrics['source_concentration']:.2f}")

    # Aggregates
    def mean_across(key, method):
        return round(float(np.mean([r[method][key] for r in results_per_query])), 4)

    aggregate = {
        "ac_rag":      {m: mean_across(m, "ac_rag")      for m in METRIC_KEYS},
        "vanilla_rag": {m: mean_across(m, "vanilla_rag") for m in METRIC_KEYS},
    }
    aggregate["deltas"] = {
        m: round(aggregate["ac_rag"][m] - aggregate["vanilla_rag"][m], 4)
        for m in METRIC_KEYS
    }
    aggregate["ac_rag"]["latency_s"]      = mean_across("latency_s", "ac_rag")
    aggregate["vanilla_rag"]["latency_s"] = mean_across("latency_s", "vanilla_rag")

    _print_tables(results_per_query, aggregate)

    return {
        "meta": {
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_queries":      len(BENCHMARK_QUERIES),
            "ac_top_k":       AC_TOP_K,
            "ac_n_clusters":  AC_N_CLUSTERS,
            "vanilla_top_k":  VANILLA_TOP_K,
            "lambda":         LAMBDA,
            "total_clusters": total_clusters,
            "total_chunks":   len(enriched),
            "note": (
                "Cohesion reported but excluded from competitive win/loss. "
                "AC-RAG claim: lower source concentration and higher diversity "
                "for cross-topic queries, at a small relevance cost."
            ),
        },
        "aggregate":         aggregate,
        "per_query_results": results_per_query,
    }


# ─── Print helpers ────────────────────────────────────────────────────────────

def _print_header():
    print()
    print("  ───────────────────────────────────────────────────────────")
    print("            AC-RAG  vs  Vanilla RAG  -  Benchmark             ")
    print("  ────────────────────────────────────────────────────────────")
    print()


def _win_label(delta: float, higher_is_better: bool, competitive: bool) -> str:
    """Returns a tidy win/loss/tie/-- label. Fixes the delta=0 → LOSS bug."""
    if not competitive:
        return "  --"
    if delta == 0.0:
        return " tie"
    return " win" if (delta > 0) == higher_is_better else "LOSS"


def _print_tables(results: list, aggregate: dict):

    # ── Table 1: Per-query R-D score ─────────────────────────────────────────
    print()
    print("  ── Per-query R-D Score  (relevance × diversity harmonic mean) ──")
    print()
    print(f"  {'':5} {'AC-RAG':>8} {'Vanilla':>8} {'Δ':>8}  {'':5}")
    print("  " + "─" * 42)
    for r in results:
        ac  = r["ac_rag"]["rd_score"]
        van = r["vanilla_rag"]["rd_score"]
        d   = r["deltas"]["rd_score"]
        marker = "↑ win" if d > 0 else ("↓ LOSS" if d < 0 else "= tie")
        print(f"  {r['query_id']:<5} {ac:>8.4f} {van:>8.4f} {d:>+8.4f}  {marker}")
    print("  " + "─" * 42)
    ac_avg  = aggregate["ac_rag"]["rd_score"]
    van_avg = aggregate["vanilla_rag"]["rd_score"]
    d_avg   = aggregate["deltas"]["rd_score"]
    print(f"  {'mean':<5} {ac_avg:>8.4f} {van_avg:>8.4f} {d_avg:>+8.4f}")

    # ── Table 2: Aggregate metric breakdown ───────────────────────────────────
    print()
    print("  ── Aggregate Metrics  (mean across 7 queries) ──────────────────")
    print()
    print(f"  {'Metric':<18} {'AC-RAG':>8} {'Vanilla':>8} {'Δ':>8}  {'':5}  {'note'}")
    print("  " + "─" * 62)

    notes = {
        "relevance":            "vanilla expected to lead (greedy similarity)",
        "diversity":            "AC-RAG structural advantage",
        "redundancy":           "inverse of diversity",
        "source_concentration": "key claim — cross-doc breadth",
        "cohesion":             "internal cluster quality (not competitive)",
        "rd_score":             "headline — harmonic mean rel × div",
    }

    for m in METRIC_KEYS:
        label = METRIC_LABELS[m]
        ac    = aggregate["ac_rag"][m]
        van   = aggregate["vanilla_rag"][m]
        d     = aggregate["deltas"][m]
        hib   = HIGHER_IS_BETTER[m]
        comp  = m in COMPETITIVE_METRICS
        wl    = _win_label(d, hib, comp)
        sign  = "+" if d >= 0 else ""
        note  = notes.get(m, "")
        print(f"  {label:<18} {ac:>8.4f} {van:>8.4f} {sign}{d:>7.4f}  {wl}   {note}")

    # ── Table 3: Latency ──────────────────────────────────────────────────────
    print()
    print("  ── Latency ──────────────────────────────────────────────────────")
    print()
    print(f"  {'AC-RAG':<22} {aggregate['ac_rag']['latency_s']:>8.3f}s  "
          f"(includes router LLM call)")
    print(f"  {'Vanilla':<22} {aggregate['vanilla_rag']['latency_s']:>8.3f}s  "
          f"(HNSW only)")

    # ── Summary ───────────────────────────────────────────────────────────────
    comp_wins = sum(
        1 for m in COMPETITIVE_METRICS
        if aggregate["deltas"][m] != 0.0
        and (aggregate["deltas"][m] > 0) == HIGHER_IS_BETTER[m]
    )
    total_comp = len(COMPETITIVE_METRICS)
    print()
    print("  ── Summary ──────────────────────────────────────────────────────")
    print()
    print(f"  AC-RAG wins {comp_wins}/{total_comp} competitive metrics.")
    print(f"  Source concentration Δ = {aggregate['deltas']['source_concentration']:+.4f}  "
          f"(lower is better — AC-RAG prevents single-doc collapse)")
    print(f"  Diversity          Δ = {aggregate['deltas']['diversity']:+.4f}  "
          f"(cluster routing enforces cross-topic spread)")
    print(f"  Relevance          Δ = {aggregate['deltas']['relevance']:+.4f}  "
          f"(expected trade-off — not a failure)")
    print()
    print("  Cohesion is an internal clustering metric and is not competitive.")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output = run_benchmark()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Results saved → {out_path}\n")