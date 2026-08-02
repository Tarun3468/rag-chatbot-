from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    """Load a PDF or text file into LangChain Document objects."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()


def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into overlapping chunks.

    chunk_overlap prevents answers that span a chunk boundary from being
    cut off - without it, retrieval quality drops on longer answers.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
