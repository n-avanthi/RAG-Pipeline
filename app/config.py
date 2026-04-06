"""
config.py
---------
Single source of truth for every constant in the AC-RAG pipeline.
Mirrors the configuration block at the top of the original Colab notebook,
adapted for a local Flask/FastAPI project layout.

All file paths are resolved relative to the project root so the app
works regardless of the working directory it is launched from.
"""

import os
import torch

from dotenv import load_dotenv
import os

load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Project root — one level above this file (i.e. rag-pipeline/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DOCS_DIR    = os.path.join(DATA_DIR, "raw_docs")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

# Ensure directories exist at import time
os.makedirs(RAW_DOCS_DIR,     exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Artefact file paths  (all land in data/preprocessed/)
# ---------------------------------------------------------------------------
CHUNKS_PATH            = os.path.join(PREPROCESSED_DIR, "chunks.json")
EMBEDDINGS_PATH        = os.path.join(PREPROCESSED_DIR, "embeddings.npy")
METADATA_PATH          = os.path.join(PREPROCESSED_DIR, "embedding_metadata.json")
EMBEDDINGS_5D_PATH     = os.path.join(PREPROCESSED_DIR, "embeddings_5d.npy")
GMM_MODEL_PATH         = os.path.join(PREPROCESSED_DIR, "gmm_model.joblib")
GMM_SCALER_PATH        = os.path.join(PREPROCESSED_DIR, "gmm_scaler.joblib")
CLUSTER_ASSIGN_PATH    = os.path.join(PREPROCESSED_DIR, "cluster_assignments.json")
ENRICHED_METADATA_PATH = os.path.join(PREPROCESSED_DIR, "final_enriched_metadata.json")
CLUSTER_PROFILES_PATH  = os.path.join(PREPROCESSED_DIR, "cluster_profiles.json")
INDEX_PATH             = os.path.join(PREPROCESSED_DIR, "ac_rag_hnsw.index")
PATH_A_MANIFEST_PATH   = os.path.join(PREPROCESSED_DIR, "path_a_manifest.json")
ROUTING_LOG_PATH       = os.path.join(PREPROCESSED_DIR, "routing_log.json")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
MODEL_NAME    = "BAAI/bge-m3"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# CHUNK_SIZE    = 800
# CHUNK_OVERLAP = 120
CHUNK_SIZE    = 600   # down from 800; equations + paragraphs are short
CHUNK_OVERLAP = 100   # down from 120
BATCH_SIZE    = 32

# ---------------------------------------------------------------------------
# UMAP  (dimensionality reduction for GMM clustering only — NOT for retrieval)
# ---------------------------------------------------------------------------
UMAP_N_COMPONENTS = 5
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42
# UMAP_N_NEIGHBORS  = 15
# UMAP_MIN_DIST     = 0.1
UMAP_N_NEIGHBORS  = 20   # up from 15; better global structure with more data
UMAP_MIN_DIST     = 0.05 # down from 0.1; tighter clusters for GMM

# ---------------------------------------------------------------------------
# GMM  (Gaussian Mixture Model — soft clustering)
# ---------------------------------------------------------------------------
# GMM_K_MIN           = 2
# GMM_K_MAX           = 20
GMM_K_MIN = 4    # up from 2; you won't get fewer than 4 meaningful clusters
GMM_K_MAX = 12   # down from 20; keeps BIC sweep fast and avoids overfitting
GMM_N_INIT          = 3
GMM_MAX_ITER        = 200
GMM_REG_COVAR       = 1e-4
GMM_RANDOM_STATE    = 42
GMM_COVARIANCE_TYPE = "full"

# Secondary cluster membership threshold:
# A chunk is a "bridge chunk" if it has >= this probability in a second cluster
SECONDARY_PROB_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# HNSW index  (FAISS)
# ---------------------------------------------------------------------------
HNSW_M          = 32    # bi-directional links per node
HNSW_EF_SEARCH  = 64   # beam width at query time
HNSW_EF_CONSTR  = 200  # beam width at build time

# ---------------------------------------------------------------------------
# Cluster profiling  (LLM call per cluster)
# ---------------------------------------------------------------------------
# PROFILE_CHUNKS_PER_CLUS = 5
# PROFILE_MAX_CHUNK_CHARS = 400
PROFILE_CHUNKS_PER_CLUS = 7   # up from 5
PROFILE_MAX_CHUNK_CHARS = 500 # up from 400; ML abstracts need more context
PROFILE_TEMPERATURE     = 0.2
PROFILE_MAX_RETRIES     = 3

# ---------------------------------------------------------------------------
# Agentic router  (LLM routing call at query time)
# ---------------------------------------------------------------------------
ROUTER_TEMPERATURE  = 0.1
ROUTER_MAX_RETRIES  = 3
ROUTER_MIN_CLUSTERS = 2
ROUTER_MAX_CLUSTERS = 3

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# RETRIEVAL_TOP_K_PER_CLUSTER = 3
RETRIEVAL_TOP_K_PER_CLUSTER = 4  # up from 3
RETRIEVAL_DIVERSITY_LAMBDA  = 0.5   # 0 = pure similarity, 1 = pure diversity

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")   # set in your environment
LLM_MODEL_NAME  = "gpt-4o-mini"                     # fast + cheap + capable


# ---------------------------------------------------------------------------
# Ollama  (commented out — kept for reference if switching back for Colab)
# ---------------------------------------------------------------------------
# OLLAMA_HOST    = "http://localhost:11434"
# LLM_MODEL_NAME = "mistral:latest"
# PROFILE_NUM_PREDICT  = 512
# PROFILE_TIMEOUT_S    = 90
# ROUTER_NUM_PREDICT   = 256
# ROUTER_TIMEOUT_S     = 60

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
TOP_REPRESENTATIVE = 5
RANDOM_SEED        = 42