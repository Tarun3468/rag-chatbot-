import requests
import streamlit as st

# Change this to your deployed Render URL after Step 16 of the deployment guide
API_URL = "http://localhost:8000"

st.title("📄 RAG Document Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
if uploaded_file and st.button("Index Document"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    with st.spinner("Indexing document..."):
        response = requests.post(f"{API_URL}/upload", files=files)
    if response.ok:
        st.success(f"Indexed {response.json()['chunks']} chunks")
    else:
        st.error(f"Upload failed: {response.text}")

st.divider()

question = st.text_input("Ask a question about the document")
if question and st.button("Get Answer"):
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", params={"question": question})
    if response.ok:
        result = response.json()
        st.write("### Answer")
        st.write(result["answer"])
        with st.expander("Sources"):
            st.json(result["sources"])
    else:
        st.error(f"Request failed: {response.text}")
