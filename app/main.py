"""
main.py
-------
Flask application entry point for the AC-RAG backend.

Key change from previous version:
  /api/process  — runs preprocess + cluster in a background thread.
                  Returns immediately with {"status": "started"}.
  /api/process/stream — SSE endpoint the frontend polls for live log lines.
  /api/process/status — returns current pipeline step + done/error state.

This fixes the single-threaded Flask freeze where UMAP/GMM/LLM profiling
blocked the entire server (including /api/health) for 10+ minutes.

All other endpoints unchanged.
"""

import json as _json
import logging
import os
import queue
import threading
import time

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app.config import (
    CLUSTER_PROFILES_PATH,
    RAW_DOCS_DIR,
)
from app.preprocess import artefacts_exist, run_preprocessing
from app.cluster_profiles import clustering_artefacts_exist, run_clustering
from app.rag_pipeline import (
    load_pipeline_state,
    is_pipeline_ready,
    route_query,
    retrieve_with_hnsw_filtered,
    assemble_rag_context,
    generate_answer,
    # check_ollama_alive,
    check_openai_alive
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

ALLOWED_EXTENSIONS = {"pdf", "txt"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Pipeline state shared across threads ─────────────────────────────────────
_pipeline_lock = threading.Lock()
_pipeline_state = {
    "running":      False,
    "step":         None,   # current step label
    "done":         False,
    "error":        None,
    "n_chunks":     None,
    "n_clusters":   None,
}
_log_queue: queue.Queue = queue.Queue(maxsize=500)


def _push_log(message: str, level: str = "info"):
    """Push a log line into the SSE queue and also log to console."""
    entry = {"ts": time.strftime("%H:%M:%S"), "level": level, "msg": message}
    try:
        _log_queue.put_nowait(entry)
    except queue.Full:
        pass  # drop oldest not newest — frontend will reconnect
    getattr(logger, level, logger.info)(message)


def _set_step(step: str):
    with _pipeline_lock:
        _pipeline_state["step"] = step
    _push_log(f"Step: {step}")


# ── Startup ───────────────────────────────────────────────────────────────────
def try_load_state():
    loaded = load_pipeline_state()
    if loaded:
        logger.info("Pipeline state loaded at startup.")
        with _pipeline_lock:
            _pipeline_state["done"] = True
    else:
        logger.info("Artefacts not yet present — upload documents first.")


with app.app_context():
    try_load_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _list_raw_docs() -> list[dict]:
    files = []
    for fname in sorted(os.listdir(RAW_DOCS_DIR)):
        fpath = os.path.join(RAW_DOCS_DIR, fname)
        if os.path.isfile(fpath) and allowed_file(fname):
            stat = os.stat(fpath)
            files.append({
                "name":       fname,
                "size_bytes": stat.st_size,
                "extension":  fname.rsplit(".", 1)[1].lower(),
            })
    return files


# ── Background pipeline worker ────────────────────────────────────────────────
def _run_pipeline_background(force_rerun: bool):
    """
    Runs preprocess → cluster in a background thread.
    Pushes log lines into _log_queue for the SSE stream.
    Updates _pipeline_state so /api/process/status stays current.
    """
    with _pipeline_lock:
        _pipeline_state.update({"running": True, "done": False, "error": None,
                                 "step": None, "n_chunks": None, "n_clusters": None})

    try:
        # ── Step 1: Preprocessing ────────────────────────────────────────────
        _set_step("Chunking documents")
        _push_log("Loading documents from raw_docs/...")
        result1 = run_preprocessing(force_rerun=force_rerun)

        if result1.get("status") == "error":
            raise RuntimeError(result1["message"])

        n_chunks = result1.get("n_chunks", "?")
        with _pipeline_lock:
            _pipeline_state["n_chunks"] = n_chunks
        _push_log(f"Preprocessing done — {n_chunks} chunks, HNSW index built.")

        # ── Step 2: Clustering ───────────────────────────────────────────────
        _set_step("Reducing dimensions (UMAP)")
        _push_log("Starting UMAP dimensionality reduction (this takes a few minutes)...")

        # Monkey-patch a progress callback into cluster_profiles so we can
        # emit log lines during the long UMAP + GMM + profiling steps.
        import app.cluster_profiles as cp

        original_fit_umap = cp.fit_umap_5d
        def _fit_umap_patched(embeddings):
            _push_log(f"UMAP: fitting on {embeddings.shape[0]} vectors → 5D...")
            result = original_fit_umap(embeddings)
            _push_log("UMAP complete.")
            _set_step("Fitting GMM")
            return result
        cp.fit_umap_5d = _fit_umap_patched

        original_bic = cp.bic_sweep
        def _bic_patched(emb_scaled):
            _push_log("BIC sweep: finding optimal number of clusters...")
            ks, bics, aics = original_bic(emb_scaled)
            import numpy as np
            best_k = ks[int(np.argmin(bics))]
            _push_log(f"BIC sweep done — optimal k={best_k}")
            return ks, bics, aics
        cp.bic_sweep = _bic_patched

        original_profile = cp.profile_all_clusters
        def _profile_patched(cluster_groups, cluster_neighbors):
            _set_step("Generating cluster profiles (LLM)")
            _push_log(f"Profiling {len(cluster_groups)} clusters with Mistral...")
            result = original_profile(cluster_groups, cluster_neighbors)
            _push_log(f"Profiling done — {len(result.get('profiles', {}))} profiles generated.")
            return result
        cp.profile_all_clusters = _profile_patched

        result2 = run_clustering(force_rerun=force_rerun)

        # Restore originals
        cp.fit_umap_5d          = original_fit_umap
        cp.bic_sweep            = original_bic
        cp.profile_all_clusters = original_profile

        if result2.get("status") == "error":
            raise RuntimeError(result2["message"])

        n_clusters = result2.get("n_clusters", "?")
        with _pipeline_lock:
            _pipeline_state["n_clusters"] = n_clusters

        _push_log(f"Clustering done — {n_clusters} clusters.")

        # ── Load into memory ─────────────────────────────────────────────────
        _set_step("Loading pipeline into memory")
        _push_log("Loading pipeline state into memory...")
        load_pipeline_state()
        _push_log("Pipeline ready. Switch to Chat tab.")

        with _pipeline_lock:
            _pipeline_state.update({"running": False, "done": True, "step": "done"})

    except Exception as e:
        msg = str(e)
        _push_log(f"ERROR: {msg}", "error")
        with _pipeline_lock:
            _pipeline_state.update({"running": False, "error": msg, "step": None})


# ── GET /api/health ───────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    docs = _list_raw_docs()
    with _pipeline_lock:
        state = dict(_pipeline_state)
    return jsonify({
        "status"          : "ok",
        # "ollama_alive"    : check_ollama_alive(),
        "openai_alive": check_openai_alive(),
        "pipeline_ready"  : is_pipeline_ready(),
        "documents_count" : len(docs),
        "pipeline_running": state["running"],
    })


# ── GET /api/documents ────────────────────────────────────────────────────────
@app.get("/api/documents")
def list_documents():
    return jsonify({"documents": _list_raw_docs()})


# ── POST /api/upload ──────────────────────────────────────────────────────────
@app.post("/api/upload")
def upload_documents():
    if "files" not in request.files:
        return error_response("No 'files' field in request.")

    uploaded = request.files.getlist("files")
    if not uploaded:
        return error_response("No files received.")

    saved, rejected = [], []

    for f in uploaded:
        if not f.filename:
            rejected.append({"name": "(empty)", "reason": "No filename"})
            continue
        fname = secure_filename(f.filename)
        if not allowed_file(fname):
            rejected.append({"name": f.filename, "reason": "Only PDF and TXT allowed"})
            continue
        dest = os.path.join(RAW_DOCS_DIR, fname)
        f.save(dest)
        size_kb = round(os.path.getsize(dest) / 1024, 1)
        saved.append({"name": fname, "size_kb": size_kb})
        logger.info("Uploaded: %s (%.1f KB)", fname, size_kb)

    return jsonify({
        "saved":       saved,
        "rejected":    rejected,
        "total_saved": len(saved),
        "documents":   _list_raw_docs(),
    })


# ── DELETE /api/documents/<filename> ─────────────────────────────────────────
@app.delete("/api/documents/<filename>")
def delete_document(filename: str):
    safe_name = secure_filename(filename)
    fpath     = os.path.join(RAW_DOCS_DIR, safe_name)
    if not os.path.exists(fpath):
        return error_response(f"File '{safe_name}' not found.", 404)
    os.remove(fpath)
    logger.info("Deleted: %s", safe_name)
    return jsonify({"deleted": safe_name, "documents": _list_raw_docs()})


# ── POST /api/process ─────────────────────────────────────────────────────────
@app.post("/api/process")
def process():
    """
    Starts the full pipeline (preprocess + cluster) in a background thread.
    Returns immediately — poll /api/process/status or stream /api/process/stream.
    """
    with _pipeline_lock:
        if _pipeline_state["running"]:
            return jsonify({"status": "already_running",
                            "message": "Pipeline is already running."})

    if not _list_raw_docs():
        return error_response("No documents found. Upload files first.", 400)

    body        = request.get_json(silent=True) or {}
    force_rerun = bool(body.get("force_rerun", True))

    # Clear the log queue
    while not _log_queue.empty():
        try: _log_queue.get_nowait()
        except queue.Empty: break

    t = threading.Thread(
        target=_run_pipeline_background,
        args=(force_rerun,),
        daemon=True,
    )
    t.start()

    return jsonify({"status": "started", "message": "Pipeline started in background."})


# ── GET /api/process/status ───────────────────────────────────────────────────
@app.get("/api/process/status")
def process_status():
    """Snapshot of the current pipeline state — for polling."""
    with _pipeline_lock:
        state = dict(_pipeline_state)
    state["pipeline_ready"] = is_pipeline_ready()
    return jsonify(state)


# ── GET /api/process/stream ───────────────────────────────────────────────────
@app.get("/api/process/stream")
def process_stream():
    """
    Server-Sent Events stream of log lines from the background pipeline.
    The frontend connects once and receives lines as they are emitted.
    """
    def generate():
        yield "data: {\"msg\": \"Connected to log stream\"}\n\n"
        while True:
            try:
                entry = _log_queue.get(timeout=30)
                yield f"data: {_json.dumps(entry)}\n\n"
            except queue.Empty:
                # Keepalive ping
                yield ": keepalive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /api/preprocess/status ────────────────────────────────────────────────
@app.get("/api/preprocess/status")
def preprocess_status():
    pre  = artefacts_exist()
    clus = clustering_artefacts_exist()
    docs = _list_raw_docs()
    return jsonify({
        "has_documents"   : len(docs) > 0,
        "documents_count" : len(docs),
        "chunks_ready"    : pre["chunks"],
        "embeddings_ready": pre["embeddings"],
        "index_ready"     : pre["index"],
        "gmm_ready"       : clus["gmm_model"],
        "profiles_ready"  : clus["profiles"],
        "enriched_ready"  : clus["enriched"],
        "all_ready"       : all(pre.values()) and all(clus.values()),
    })


# ── GET /api/clusters ─────────────────────────────────────────────────────────
@app.get("/api/clusters")
def get_clusters():
    if not is_pipeline_ready():
        return error_response("Pipeline not ready. Process your documents first.", 503)
    with open(CLUSTER_PROFILES_PATH, "r", encoding="utf-8") as f:
        data = _json.load(f)
    profiles_raw = data.get("profiles", {})
    profiles_out = []
    for cid_str in sorted(profiles_raw.keys(), key=int):
        entry   = profiles_raw[cid_str]
        profile = entry["profile"]
        profiles_out.append({
            "cluster_id"      : int(cid_str),
            "theme"           : profile["primary_theme"],
            "key_entities"    : profile["key_entities"],
            "contrastive_edge": profile["contrastive_edge"],
            "n_members"       : entry["n_members"],
            "avg_prob"        : entry["avg_prob"],
        })
    return jsonify({"n_clusters": len(profiles_out), "profiles": profiles_out})


# ── POST /api/query ───────────────────────────────────────────────────────────
@app.post("/api/query")
def query():
    if not is_pipeline_ready():
        return error_response(
            "Pipeline not ready. Upload and process your documents first.", 503)

    body = request.get_json(silent=True)
    if not body or not body.get("query"):
        return error_response("Request body must include 'query'.")

    user_query       = str(body["query"]).strip()
    top_k            = int(body.get("top_k", 3))
    diversity_lambda = float(body.get("diversity_lambda", 0.5))
    do_generate      = bool(body.get("generate_answer", True))

    if len(user_query) < 3:
        return error_response("Query must be at least 3 characters.")

    try:
        routing     = route_query(user_query)
        retrieved   = retrieve_with_hnsw_filtered(
            routing_result=routing, top_k=top_k, diversity_lambda=diversity_lambda)
        rag_context = assemble_rag_context(retrieved, routing)

        answer, answer_latency = None, None
        if do_generate:
            answer, answer_latency = generate_answer(user_query, rag_context)

        from app.rag_pipeline import _cluster_profiles
        cluster_results = []
        for cid, chunks in retrieved.items():
            theme = _cluster_profiles.get(str(cid), {}).get(
                "profile", {}).get("primary_theme", f"Cluster {cid}")
            cluster_results.append({
                "cluster_id": cid,
                "theme"     : theme,
                "chunks"    : [{
                    "chunk_id"           : c["chunk_id"],
                    "source_doc"         : os.path.basename(c["source_doc"]),
                    "text"               : c["text"],
                    "similarity_score"   : c.get("similarity_score", c["primary_probability"]),
                    "primary_cluster"    : c["primary_cluster"],
                    "primary_probability": c["primary_probability"],
                    "is_bridge_chunk"    : c["is_bridge_chunk"],
                } for c in chunks],
            })

        return jsonify({
            "query"           : user_query,
            "routing"         : {
                "selected_clusters": routing["selected_clusters"],
                "reasoning"        : routing["reasoning"],
                "cluster_themes"   : routing["cluster_themes"],
                "latency_s"        : routing["llm_meta"]["elapsed_s"],
            },
            "results"         : cluster_results,
            "rag_context"     : rag_context,
            "answer"          : answer,
            "answer_latency_s": answer_latency,
            "total_chunks"    : sum(len(v) for v in retrieved.values()),
        })

    except RuntimeError as e:
        logger.error("Query failed: %s", e)
        return error_response(str(e), 500)
    except Exception as e:
        logger.exception("Unexpected error during query")
        return error_response(f"Internal error: {e}", 500)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=True is essential — allows health/stream endpoints to respond
    # while the pipeline runs in its background thread
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)