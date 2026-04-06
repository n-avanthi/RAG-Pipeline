# AC-RAG — Adaptive Cluster Retrieval

Three-column web interface for the AC-RAG pipeline.
Backend: Flask · Frontend: React + Vite

---

## Folder structure

```
rag-pipeline/
├── app/
│   ├── __init__.py
│   ├── config.py            # all constants (paths, model names, hyperparams)
│   ├── main.py              # Flask app + all API routes
│   ├── preprocess.py        # load → chunk → embed → HNSW
│   ├── cluster_profiles.py  # UMAP → GMM → soft assignments → LLM profiling
│   ├── rag_pipeline.py      # router → HNSW retrieval → MMR → answer
│   └── models/
│       └── schemas.py       # Pydantic schemas
├── data/
│   ├── raw_docs/            # put your .pdf and .txt files here
│   └── preprocessed/        # generated artefacts land here automatically
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           ├── QueryPanel.jsx
│           ├── RoutingPanel.jsx
│           └── ResultsPanel.jsx
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python backend

```bash
# from rag-pipeline/
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama (local LLM)

```bash
# Install from https://ollama.com
ollama pull mistral:latest
ollama serve                     # runs on http://localhost:11434
```

### 3. React frontend

```bash
cd frontend
npm install
```

---

## Running

### Backend

```bash
# from rag-pipeline/
python -m app.main
# → Flask on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm run dev
# → Vite on http://localhost:5173
```

---

## Pipeline steps (in order)

1. **Drop documents** into `data/raw_docs/` (.pdf or .txt)

2. **Preprocess** — POST `/api/preprocess`
   - Loads documents, chunks them, embeds with BGE-M3, builds HNSW index
   - Or click **"1 · Preprocess"** in the UI header

3. **Cluster** — POST `/api/cluster`
   - UMAP 1024D→5D, BIC sweep to find optimal k, final GMM,
     soft assignments, enriched metadata, two-pass contrastive LLM profiling
   - Or click **"2 · Cluster"** in the UI header

4. **Query** — POST `/api/query`
   - Mistral routes to 2-3 clusters → HNSW filtered retrieval + MMR → Mistral answer

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + Ollama + pipeline status |
| GET | `/api/preprocess/status` | Which artefacts exist on disk |
| POST | `/api/preprocess` | Run preprocessing pipeline |
| POST | `/api/cluster` | Run clustering + LLM profiling |
| GET | `/api/clusters` | Return all cluster profiles |
| POST | `/api/query` | Full query pipeline |

### POST /api/query — request body

```json
{
  "query": "What are the economic impacts of EV batteries?",
  "top_k": 3,
  "diversity_lambda": 0.5,
  "generate_answer": true
}
```

---

## Key design decisions

- **config.py** is the single source of truth — all hyperparameters live there
- **UMAP vectors are temporary** — used only for GMM clustering, never for retrieval
- **Retrieval uses raw 1024D BGE-M3 vectors** in HNSW (cosine via inner product on unit vecs)
- **MMR reranking** within each cluster balances similarity vs diversity
- **Two-pass profiling** injects nearest-neighbor profiles into pass-2 prompts
- Module-level singletons (`_bge_model`, `_hnsw_index`, `_enriched`) are loaded once at startup