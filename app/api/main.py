import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from app.ingestion.embed import build_vectorstore
from app.ingestion.loader import chunk_documents, load_document
from app.retrieval.chain import answer_question

load_dotenv()

app = FastAPI(title="RAG Chatbot API")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save an uploaded document, chunk it, and index it in the vector store."""
    save_path = f"data/uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_document(save_path)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return {"status": "indexed", "chunks": len(chunks)}


@app.post("/ask")
async def ask(question: str):
    """Answer a question using the indexed documents."""
    result = answer_question(question)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
