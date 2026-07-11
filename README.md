# Retrieval-Augmented Generation using Semantic Clustering

A Retrieval-Augmented Generation (RAG) system that improves retrieval diversity by organizing document embeddings into semantic clusters before retrieval. Instead of relying solely on similarity-based top-*k* retrieval, the system routes queries through relevant semantic clusters to reduce redundancy and improve context coverage, particularly for multi-topic queries.

## Features

- Semantic document retrieval using **BAAI/bge-m3** embeddings
- Approximate nearest neighbor search with **FAISS (HNSW)**
- Semantic clustering using **UMAP** and **Gaussian Mixture Models (GMM)**
- LLM-generated cluster profiles for query routing
- Maximal Marginal Relevance (MMR) reranking for diverse retrieval
- Interactive web interface built with React and Flask

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Backend | Flask |
| Frontend | React + Vite |
| Embedding Model | BAAI/bge-m3 |
| LLM | OpenAI GPT-4o-mini |
| Vector Index | FAISS (HNSW) |
| Clustering | UMAP + Gaussian Mixture Model (GMM) |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API key

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Backend Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m app.main
```

The backend will be available at:

```
http://localhost:8000
```

---

# Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```
http://localhost:5173
```

---

# Using the Application

### 1. Add Documents

Place PDF or text documents inside:

```
data/raw_docs/
```

---

### 2. Preprocess Documents

Run the preprocessing pipeline to:

- Load documents
- Split documents into chunks
- Generate embeddings
- Build the FAISS vector index

API:

```
POST /api/preprocess
```

or click **Preprocess** from the web interface.

---

### 3. Generate Semantic Clusters

Run the clustering pipeline to:

- Reduce embedding dimensions using UMAP
- Generate semantic clusters using GMM
- Create cluster profiles for query routing

API:

```
POST /api/cluster
```

or click **Cluster** from the web interface.

---

### 4. Query the System

Submit a query through the UI or API.

The system will:

1. Identify relevant semantic clusters
2. Retrieve candidate document chunks
3. Apply MMR reranking
4. Generate a response using GPT-4o-mini

Example request:

```json
{
  "query": "What are the economic impacts of EV batteries?",
  "top_k": 3,
  "diversity_lambda": 0.5,
  "generate_answer": true
}
```

API:

```
POST /api/query
```

---

# API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/preprocess/status` | View preprocessing status |
| POST | `/api/preprocess` | Run preprocessing pipeline |
| POST | `/api/cluster` | Generate semantic clusters |
| GET | `/api/clusters` | Retrieve cluster profiles |
| POST | `/api/query` | Execute the retrieval and generation pipeline |

---

# Project Configuration

Project-wide settings such as embedding model, clustering parameters, retrieval settings, and generation options are managed through:

```
config.py
```

---

# Technologies

- Flask
- React
- Vite
- OpenAI GPT-4o-mini
- BAAI/bge-m3
- FAISS (HNSW)
- UMAP
- Gaussian Mixture Model (GMM)

---

# License

This project is intended for academic and research purposes.
