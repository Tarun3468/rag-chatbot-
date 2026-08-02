"""
Quick sanity check for the ingestion pipeline.
Run this after Phase 2 (Step 7) to confirm chunking + embedding works
before wiring up the API and frontend.

Usage: python test_ingest.py path/to/your/document.pdf
"""
import sys

from app.ingestion.embed import build_vectorstore
from app.ingestion.loader import chunk_documents, load_document

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ingest.py path/to/document.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks from {file_path}")
