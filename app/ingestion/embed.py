from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# all-MiniLM-L6-v2 is small, fast, and free to run locally - good default
# for a portfolio project since it doesn't burn API credits on every chunk.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(chunks, persist_directory: str = "data/chroma_db"):
    """Embed chunks and persist them to a local Chroma vector store."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore(persist_directory: str = "data/chroma_db"):
    """Load an existing Chroma vector store from disk."""
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
