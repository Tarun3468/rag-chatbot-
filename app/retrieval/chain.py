from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from app.ingestion.embed import load_vectorstore


def get_qa_chain():
    """Build a simple vector-only RetrievalQA chain."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def answer_question(question: str):
    """Run a question through the QA chain and return answer + sources."""
    chain = get_qa_chain()
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }
