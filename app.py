# app.py
import streamlit as st
from embedding_HF import process_source, initialize_qa_system, answer_question
import os

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Chat with Docs", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.title(" Document / URL Input")
source_type = st.sidebar.radio("Choose source type:", ["Upload File", "Enter URL"])

if source_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your document", type=["docx", "txt", "pdf"])
    if uploaded_file:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        filename = f"uploaded.{file_extension}"
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source = filename
        st.sidebar.success(f" Uploaded: {uploaded_file.name}")
    else:
        source = None
else:
    source = st.sidebar.text_input("Enter a webpage URL")
    if source:
        st.sidebar.info(" URL entered")

process_btn = st.sidebar.button("Process Source")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()

# ---------------- MAIN AREA ----------------
st.title(" Chat with Your Document / URL")
st.write("Ask questions based on the uploaded file or entered URL. Supports DOCX, TXT, PDF files and web URLs!")

# Supported formats info
with st.expander(" Supported Formats"):
    st.markdown("""
    - ** DOCX** - Microsoft Word documents
    - ** TXT** - Plain text files  
    - ** PDF** - Portable Document Format
    - ** URLs** - Web pages and online content
    """)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_processed" not in st.session_state:
    st.session_state.source_processed = False

# Process document
if process_btn and source:
    with st.spinner("Processing source..."):
        try:
            # Remove existing chunks file to force reprocessing
            if os.path.exists("chunks.pkl"):
                os.remove("chunks.pkl")
                
            process_source(source)
            st.session_state.model, st.session_state.index, st.session_state.chunks = initialize_qa_system()
            st.session_state.source_processed = True
            st.success("Source processed successfully! Start chatting below.")
            
            # Show chunk info
            if st.session_state.chunks:
                st.info(f"Processed {len(st.session_state.chunks)} chunks from the source")
                
        except Exception as e:
            st.error(f"Error processing source: {str(e)}")
            st.session_state.source_processed = False

# ---------------- CHAT INTERFACE ----------------
if st.session_state.model is not None and st.session_state.source_processed:
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask a question about your document...")
    
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        # Generate and display assistant response
        with st.spinner("Generating answer..."):
            try:
                answer = answer_question(query, st.session_state.model, st.session_state.index, st.session_state.chunks)
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").markdown(answer)
                
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").markdown(error_msg)

else:
    if not st.session_state.source_processed:
        st.info("Please upload a document or enter a URL from the sidebar and click 'Process Source' to begin.")
    else:
        st.warning("Please process a source document first to start chatting.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • Supports DOCX, TXT, PDF & URLs"
    "</div>",
    unsafe_allow_html=True
)