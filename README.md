# RAG Document Q&A Chatbot

A chatbot that answers questions grounded in your own uploaded documents (PDF/text),
using retrieval-augmented generation with hybrid search.

## Problem
Users need to quickly extract answers from long documents without reading the whole thing.

## Architecture
```
User → Streamlit Frontend → FastAPI Backend → Retrieval (Chroma + BM25) → LLM → Answer
                                    ↑
                          Document Ingestion Pipeline
```

## Tech Stack
- LangChain (orchestration)
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- BM25 (keyword retrieval, via `rank_bm25`)
- OpenAI GPT-4o-mini (generation)
- FastAPI (backend)
- Streamlit (frontend)
- RAGAS (evaluation)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run locally
```bash
# Terminal 1
uvicorn app.api.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

## Evaluation Results
<!-- Fill in after running evaluate.py -->
| Metric | Vector-only | Hybrid (BM25 + vector) |
|---|---|---|
| Faithfulness | | |
| Answer Relevancy | | |
| Context Precision | | |

## Live Demo
<!-- Add your deployed Streamlit Cloud link here -->

## Known Limitations
- Render free tier spins down after inactivity - first request after idle may take ~30s
- ChromaDB storage is ephemeral on Render's free tier (fine for demo purposes)
