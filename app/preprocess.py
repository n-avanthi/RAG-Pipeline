"""
preprocess.py
-------------
Offline pipeline — Step 1 of the AC-RAG system.

Functions here mirror the original Colab notebook exactly:
  load_documents_from_folder()
  chunk_documents()
  embed_chunks()
  build_metadata()
  build_hnsw_index()
  verify_embeddings()
  run_preprocessing()   ← orchestrator called by the Flask endpoint

All heavy objects (model, index) are kept module-level so they are
loaded once per process, not once per request.
"""

import os
import gc
import json
import time
import logging

import numpy as np
import numpy.linalg as LA
import torch
import faiss
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import (
    MODEL_NAME, DEVICE, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
    HNSW_M, HNSW_EF_SEARCH, HNSW_EF_CONSTR,
    RAW_DOCS_DIR,
    CHUNKS_PATH, EMBEDDINGS_PATH, METADATA_PATH, INDEX_PATH,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused by retrieval
# ---------------------------------------------------------------------------
_bge_model:  SentenceTransformer | None = None
_hnsw_index: faiss.IndexHNSWFlat  | None = None


def get_bge_model() -> SentenceTransformer:
    global _bge_model
    if _bge_model is None:
        logger.info("Loading BGE-M3 onto %s …", DEVICE)
        _bge_model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
        )
        _bge_model.eval()
        logger.info("BGE-M3 loaded.")
    return _bge_model


def get_hnsw_index() -> faiss.IndexHNSWFlat | None:
    global _hnsw_index
    if _hnsw_index is None and os.path.exists(INDEX_PATH):
        logger.info("Loading HNSW index from %s …", INDEX_PATH)
        _hnsw_index = faiss.read_index(INDEX_PATH)
        logger.info("HNSW index loaded. Vectors: %d", _hnsw_index.ntotal)
    return _hnsw_index


# ---------------------------------------------------------------------------
# 1. Document loading
# ---------------------------------------------------------------------------

def load_documents_from_folder(folder_path: str) -> list:
    """
    Recursively scans folder_path for .pdf and .txt files.
    Returns a list of LangChain Document objects with metadata.
    Identical to the Colab version.
    """
    raw_docs    = []
    found_files = []

    for root, _, files in os.walk(folder_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext   = fname.lower().split(".")[-1]

            try:
                if ext == "pdf":
                    loader = PyMuPDFLoader(fpath)
                    docs   = loader.load()
                    raw_docs.extend(docs)
                    found_files.append(("PDF", fpath, len(docs)))

                elif ext == "txt":
                    try:
                        loader = TextLoader(fpath, encoding="utf-8")
                        docs   = loader.load()
                    except UnicodeDecodeError:
                        loader = TextLoader(fpath, encoding="latin-1")
                        docs   = loader.load()
                    raw_docs.extend(docs)
                    found_files.append(("TXT", fpath, len(docs)))

            except Exception as e:
                logger.warning("Skipping %s: %s", fname, e)

    logger.info("Scanned %s — %d files, %d LangChain docs",
                folder_path, len(found_files), len(raw_docs))
    return raw_docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    raw_docs: list,
    chunk_size: int    = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Splits raw LangChain Documents into fixed-size chunks.
    Returns list of dicts: {chunk_id, source_doc, text}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = chunk_overlap,
        length_function = len,
        add_start_index = True,
    )

    chunks      = []
    chunk_index = {}
    split_docs  = splitter.split_documents(raw_docs)

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        base   = os.path.splitext(os.path.basename(source))[0]

        chunk_index[base] = chunk_index.get(base, 0)
        cid = f"{base}_chunk_{chunk_index[base]:04d}"
        chunk_index[base] += 1

        chunks.append({
            "chunk_id"  : cid,
            "source_doc": source,
            "text"      : doc.page_content.strip(),
        })

    logger.info("Chunking complete — %d raw docs → %d chunks", len(raw_docs), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks    : list[dict],
    model     : SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Encodes all chunks in mini-batches using BGE-M3.
    Returns float32 array of shape (N, 1024).
    Vectors are L2-normalised (cosine-ready).
    """
    texts      = [c["text"] for c in chunks]
    n          = len(texts)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    logger.info("Embedding %d chunks …", n)

    for start in tqdm(range(0, n, batch_size), desc="Embedding"):
        end   = min(start + batch_size, n)
        batch = texts[start:end]

        with torch.no_grad():
            batch_emb = model.encode(
                batch,
                batch_size           = batch_size,
                convert_to_numpy     = True,
                normalize_embeddings = True,
                show_progress_bar    = False,
            )

        embeddings[start:end] = batch_emb

    logger.info("Embedding complete. Shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# 4. Metadata
# ---------------------------------------------------------------------------

def build_metadata(chunks: list[dict]) -> list[dict]:
    """
    Creates an index-aligned metadata list:
    metadata[i] ↔ embeddings[i] ↔ chunks[i]
    """
    return [
        {
            "index"     : i,
            "chunk_id"  : c["chunk_id"],
            "source_doc": c["source_doc"],
            "text"      : c["text"],
        }
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# 5. HNSW index
# ---------------------------------------------------------------------------

def build_hnsw_index(
    embeddings: np.ndarray,
    d:          int = EMBEDDING_DIM,
    M:          int = HNSW_M,
    ef_constr:  int = HNSW_EF_CONSTR,
    ef_search:  int = HNSW_EF_SEARCH,
) -> faiss.IndexHNSWFlat:
    """
    Constructs a FAISS HNSW graph index on unit-normalised vectors.
    Uses METRIC_INNER_PRODUCT — equivalent to cosine on unit vecs.
    """
    logger.info("Building IndexHNSWFlat d=%d M=%d …", d, M)
    t0 = time.perf_counter()

    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_constr
    index.hnsw.efSearch        = ef_search
    index.add(embeddings)

    elapsed = time.perf_counter() - t0
    logger.info("HNSW built in %.2fs — %d vectors", elapsed, index.ntotal)
    return index


# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------

def verify_embeddings(embeddings: np.ndarray, metadata: list[dict]) -> dict:
    """
    Sanity checks on the embedding matrix.
    Returns a dict of diagnostics (logged + returned for the API response).
    """
    n, dim  = embeddings.shape
    norms   = LA.norm(embeddings, axis=1)
    are_unit = np.allclose(norms, 1.0, atol=1e-3)
    aligned  = n == len(metadata)

    diagnostics = {
        "shape"       : list(embeddings.shape),
        "dtype"       : str(embeddings.dtype),
        "unit_normed" : bool(are_unit),
        "aligned"     : aligned,
        "norm_mean"   : float(norms.mean()),
        "norm_std"    : float(norms.std()),
        "has_nan"     : bool(np.isnan(embeddings).any()),
        "has_inf"     : bool(np.isinf(embeddings).any()),
        "zero_rows"   : int(np.all(embeddings == 0, axis=1).sum()),
    }

    if not aligned:
        logger.error("Alignment mismatch: %d embeddings vs %d metadata records", n, len(metadata))
    if diagnostics["has_nan"] or diagnostics["has_inf"]:
        logger.error("Embeddings contain NaN or Inf — check fp16 overflow")

    return diagnostics


# ---------------------------------------------------------------------------
# 7. Persistence helpers
# ---------------------------------------------------------------------------

def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON → %s  (%.2f MB)", path, os.path.getsize(path) / 1024**2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 8. Orchestrator — called by Flask endpoint
# ---------------------------------------------------------------------------

def artefacts_exist() -> dict[str, bool]:
    """Returns a dict of which artefacts are already on disk."""
    return {
        "chunks"     : os.path.exists(CHUNKS_PATH),
        "embeddings" : os.path.exists(EMBEDDINGS_PATH),
        "metadata"   : os.path.exists(METADATA_PATH),
        "index"      : os.path.exists(INDEX_PATH),
    }


def run_preprocessing(force_rerun: bool = False) -> dict:
    """
    Full offline preprocessing pipeline.
    Idempotent: skips steps whose output already exists on disk
    unless force_rerun=True.

    Returns a summary dict consumed by the API response.
    """
    existing = artefacts_exist()
    all_done = all(existing.values())

    if all_done and not force_rerun:
        logger.info("All preprocessing artefacts already exist. Skipping.")
        chunks = load_json(CHUNKS_PATH)
        return {
            "status"   : "already_done",
            "message"  : "All artefacts present on disk. Pass force_rerun=true to re-run.",
            "n_chunks" : len(chunks),
        }

    t_total = time.perf_counter()

    # Step 1: Load
    raw_docs = load_documents_from_folder(RAW_DOCS_DIR)
    if not raw_docs:
        return {
            "status" : "error",
            "message": f"No .pdf or .txt files found in {RAW_DOCS_DIR}",
        }

    # Step 2: Chunk
    chunks = chunk_documents(raw_docs)
    save_json(chunks, CHUNKS_PATH)

    # Step 3: Embed
    model      = get_bge_model()
    embeddings = embed_chunks(chunks, model)

    # Step 4: Metadata
    metadata = build_metadata(chunks)

    # Step 5: Save embeddings + metadata
    np.save(EMBEDDINGS_PATH, embeddings)
    save_json(metadata, METADATA_PATH)
    logger.info("Embeddings saved → %s", EMBEDDINGS_PATH)

    # Step 6: Verify
    diagnostics = verify_embeddings(embeddings, metadata)
    if diagnostics["has_nan"] or diagnostics["has_inf"]:
        return {
            "status" : "error",
            "message": "Embedding verification failed — NaN or Inf detected.",
            "diagnostics": diagnostics,
        }

    # Step 7: HNSW
    global _hnsw_index
    _hnsw_index = build_hnsw_index(embeddings)
    faiss.write_index(_hnsw_index, INDEX_PATH)
    logger.info("HNSW index saved → %s", INDEX_PATH)

    # Free GPU memory
    del embeddings
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_total
    logger.info("Preprocessing complete in %.1fs", elapsed)

    return {
        "status"      : "done",
        "message"     : f"Preprocessing complete in {elapsed:.1f}s",
        "n_chunks"    : len(chunks),
        "diagnostics" : diagnostics,
        "elapsed_s"   : round(elapsed, 2),
    }