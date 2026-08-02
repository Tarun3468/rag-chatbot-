from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from app.ingestion.embed import load_vectorstore


def get_hybrid_retriever(vectorstore, chunks):
    """
    Combine keyword search (BM25) with semantic search (vector) into one
    retriever. Pure semantic search misses exact keyword matches like
    names, IDs, or codes - BM25 catches those. Weighting favors semantic
    search slightly (0.6) since it generalizes better across phrasing.
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble_retriever


def get_hybrid_qa_chain(chunks):
    """Build a RetrievalQA chain using the hybrid retriever."""
    vectorstore = load_vectorstore()
    retriever = get_hybrid_retriever(vectorstore, chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def answer_question_hybrid(question: str, chunks):
    """Run a question through the hybrid QA chain and return answer + sources."""
    chain = get_hybrid_qa_chain(chunks)
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }
