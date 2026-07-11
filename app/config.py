import os
import torch
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
RAW_DOCS_DIR     = os.path.join(DATA_DIR, "raw_docs")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

os.makedirs(RAW_DOCS_DIR,     exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

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

MODEL_NAME    = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
BATCH_SIZE    = 32

UMAP_N_COMPONENTS = 5
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42
UMAP_N_NEIGHBORS  = 20
UMAP_MIN_DIST     = 0.05

GMM_K_MIN           = 4
GMM_K_MAX           = 12
GMM_N_INIT          = 3
GMM_MAX_ITER        = 200
GMM_REG_COVAR       = 1e-4
GMM_RANDOM_STATE    = 42
GMM_COVARIANCE_TYPE = "full"

SECONDARY_PROB_THRESHOLD = 0.20

HNSW_M         = 32
HNSW_EF_SEARCH = 64
HNSW_EF_CONSTR = 200

PROFILE_CHUNKS_PER_CLUS = 7
PROFILE_MAX_CHUNK_CHARS = 500
PROFILE_TEMPERATURE     = 0.2
PROFILE_MAX_RETRIES     = 3

ROUTER_TEMPERATURE  = 0.1
ROUTER_MAX_RETRIES  = 3
ROUTER_MIN_CLUSTERS = 2
ROUTER_MAX_CLUSTERS = 3

RETRIEVAL_TOP_K_PER_CLUSTER = 4
RETRIEVAL_DIVERSITY_LAMBDA  = 0.5

LLM_MODEL_NAME = "gpt-4o-mini"

RANDOM_SEED = 42