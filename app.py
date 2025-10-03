# app.py
import streamlit as st
from embedding import process_source, initialize_qa_system, answer_question
import os
import fastapi
# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Chat with Docs", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.title("📂 Document / URL Input")
source_type = st.sidebar.radio("Choose source type:", ["Upload .docx", "Enter URL"])

if source_type == "Upload .docx":
    uploaded_file = st.sidebar.file_uploader("Upload your .docx file", type=["docx"])
    if uploaded_file:
        with open("uploaded.docx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        source = "uploaded.docx"
    else:
        source = None
else:
    source = st.sidebar.text_input("Enter a webpage URL")

process_btn = st.sidebar.button("Process Source")

# ---------------- MAIN AREA ----------------
st.title("💬 Chat with Your Document / URL")
st.write("Ask questions based on the uploaded file or entered URL. Works like Chat-with-PDF/Docs!")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process document
if process_btn and source:
    with st.spinner("Processing source..."):
        process_source(source)
        st.session_state.model, st.session_state.index, st.session_state.chunks = initialize_qa_system()
    st.success("✅ Source processed successfully! Start chatting below.")

# ---------------- CHAT INTERFACE ----------------
if st.session_state.model is not None:
    # Chat container
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Input box
    query = st.chat_input("👉 Ask a question")
    if query:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner(" Generating answer..."):
            answer = answer_question(query, st.session_state.model, st.session_state.index, st.session_state.chunks)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

else:
    st.info("ℹ️ Please upload a document or enter a URL from the sidebar to begin.")
