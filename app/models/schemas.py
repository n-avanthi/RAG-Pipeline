"""
schemas.py
----------
Pydantic models for all FastAPI request bodies and response payloads.
Keeps main.py clean and gives automatic OpenAPI docs for free.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Preprocessing endpoints
# ---------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    """
    POST /api/preprocess
    Triggers the full offline pipeline:
      load docs → chunk → embed → HNSW index → UMAP → GMM → cluster profiles
    """
    force_rerun: bool = Field(
        default=False,
        description="If True, re-runs even if preprocessed artefacts already exist on disk."
    )


class PreprocessStatus(BaseModel):
    """
    GET /api/preprocess/status
    Reports whether each artefact exists on disk.
    """
    chunks_ready:       bool
    embeddings_ready:   bool
    index_ready:        bool
    gmm_ready:          bool
    profiles_ready:     bool
    enriched_ready:     bool
    all_ready:          bool


class PreprocessResponse(BaseModel):
    status:       str           # "started" | "already_done" | "error"
    message:      str
    n_chunks:     Optional[int] = None
    n_clusters:   Optional[int] = None


# ---------------------------------------------------------------------------
# Query / retrieval endpoints
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    POST /api/query
    Full pipeline: route → retrieve → assemble context → generate answer.
    """
    query: str = Field(..., min_length=3, description="Natural language question.")
    top_k: int = Field(
        default=3,
        ge=1, le=10,
        description="Chunks to retrieve per selected cluster."
    )
    diversity_lambda: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="MMR diversity weight. 0 = pure similarity, 1 = pure diversity."
    )
    generate_answer: bool = Field(
        default=True,
        description="If True, passes assembled context to Mistral for a final answer."
    )


class ClusterTheme(BaseModel):
    cluster_id:        int
    theme:             str
    contrastive_edge:  str
    n_members:         int
    avg_prob:          float


class RetrievedChunk(BaseModel):
    chunk_id:            str
    source_doc:          str
    text:                str
    similarity_score:    float
    primary_cluster:     int
    primary_probability: float
    is_bridge_chunk:     bool


class ClusterResult(BaseModel):
    cluster_id:    int
    theme:         str
    chunks:        list[RetrievedChunk]


class RoutingResult(BaseModel):
    selected_clusters: list[int]
    reasoning:         str
    cluster_themes:    dict[str, str]   # cluster_id (str) → theme
    latency_s:         float


class QueryResponse(BaseModel):
    query:           str
    routing:         RoutingResult
    results:         list[ClusterResult]
    rag_context:     str
    answer:          Optional[str] = None
    answer_latency_s: Optional[float] = None
    total_chunks:    int


# ---------------------------------------------------------------------------
# Cluster info endpoints
# ---------------------------------------------------------------------------

class ClusterProfileResponse(BaseModel):
    """
    GET /api/clusters
    Returns all cluster profiles for display in the frontend routing panel.
    """
    n_clusters:  int
    profiles:    list[ClusterTheme]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:        str   # "ok"
    ollama_alive:  bool
    pipeline_ready: bool