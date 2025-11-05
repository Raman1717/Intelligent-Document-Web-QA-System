import re 
import os 
import pickle 
import requests 
import json 
import faiss 
import numpy as np 
import logging 
from docx import Document 
from typing import List, Tuple, Optional, Union, Dict, Any
from sentence_transformers import SentenceTransformer 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from bs4 import BeautifulSoup 
from urllib.parse import urlparse 
import PyPDF2
import fitz  # PyMuPDF
import mysql.connector
from mysql.connector import Error
import uuid
from datetime import datetime

# ------------------ NLTK DOWNLOAD CHECK ------------------ 
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration 
API_KEY = "AIzaSyDL4T66vw6uN0UgsGBxxuTFqVE9Nes84sQ"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
DEFAULT_TOP_K = 3

# Database configuration - UPDATE THESE WITH YOUR MYSQL CREDENTIALS
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Raman@mysql',  # Change this to your MySQL password
    'database': 'rag_chat_system'
}

# ------------------ Database Functions ------------------

def get_db_connection():
    """Create and return database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

def create_chat_session(document_source: str, chunks: List[str]) -> str:
    """Create a new chat session and store chunks in database"""
    session_id = str(uuid.uuid4())
    connection = get_db_connection()
    if not connection:
        logger.error("Failed to connect to database")
        return None
    
    try:
        cursor = connection.cursor()
        
        # Insert session
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, document_source, chunk_count) VALUES (%s, %s, %s)",
            (session_id, document_source, len(chunks))
        )
        
        # Insert chunks
        chunk_data = [(session_id, i, chunk) for i, chunk in enumerate(chunks)]
        cursor.executemany(
            "INSERT INTO session_chunks (session_id, chunk_index, chunk_text) VALUES (%s, %s, %s)",
            chunk_data
        )
        
        connection.commit()
        logger.info(f"Created chat session {session_id} with {len(chunks)} chunks")
        return session_id
        
    except Error as e:
        logger.error(f"Error creating chat session: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def save_chat_message(session_id: str, message_type: str, content: str, retrieved_chunks: List[Dict] = None):
    """Save a chat message to database"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        # Convert retrieved chunks to JSON
        chunks_json = json.dumps(retrieved_chunks) if retrieved_chunks else None
        
        cursor.execute(
            "INSERT INTO chat_messages (session_id, message_type, content, retrieved_chunks) VALUES (%s, %s, %s, %s)",
            (session_id, message_type, content, chunks_json)
        )
        
        connection.commit()
        logger.info(f"Saved {message_type} message to session {session_id}")
        return True
        
    except Error as e:
        logger.error(f"Error saving chat message: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_chat_sessions() -> List[Dict]:
    """Get all chat sessions ordered by latest first"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT session_id, document_source, chunk_count, created_at, updated_at "
            "FROM chat_sessions ORDER BY updated_at DESC"
        )
        sessions = cursor.fetchall()
        
        # Convert datetime objects to strings for JSON serialization
        for session in sessions:
            session['created_at'] = session['created_at'].isoformat() if session['created_at'] else None
            session['updated_at'] = session['updated_at'].isoformat() if session['updated_at'] else None
            
        return sessions
        
    except Error as e:
        logger.error(f"Error fetching chat sessions: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_chat_history(session_id: str) -> List[Dict]:
    """Get chat history for a specific session"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT message_type, content, retrieved_chunks, created_at "
            "FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,)
        )
        messages = cursor.fetchall()
        
        # Parse JSON fields and convert datetime
        for message in messages:
            if message['retrieved_chunks']:
                message['retrieved_chunks'] = json.loads(message['retrieved_chunks'])
            message['created_at'] = message['created_at'].isoformat() if message['created_at'] else None
            
        return messages
        
    except Error as e:
        logger.error(f"Error fetching chat history: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_session_chunks(session_id: str) -> List[str]:
    """Get chunks for a specific session"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT chunk_text FROM session_chunks WHERE session_id = %s ORDER BY chunk_index ASC",
            (session_id,)
        )
        chunks = [row[0] for row in cursor.fetchall()]
        return chunks
        
    except Error as e:
        logger.error(f"Error fetching session chunks: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def delete_chat_session(session_id: str) -> bool:
    """Delete a chat session and all related data"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        connection.commit()
        logger.info(f"Deleted chat session {session_id}")
        return True
        
    except Error as e:
        logger.error(f"Error deleting chat session: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# ------------------- Phase 1 — Document Ingestion & Chunking ----------------------------

def process_source(source: str, session_id: str = None) -> Tuple[List[str], str]:
    """ 
    Complete source processing pipeline for either file or URL.
    Returns chunks and session_id.
    """
    print(" Processing document and URL ...")
    chunks = extract_chunks(source)
    
    # Create new session if not provided
    if not session_id:
        session_id = create_chat_session(source, chunks)
        if not session_id:
            logger.error("Failed to create chat session, using local storage only")
    
    save_chunks_to_file(chunks)
    print(f"Extracted {len(chunks)} chunks and saved to chunks.pkl")
    if session_id:
        print(f"Session ID: {session_id}")
    return chunks, session_id

def extract_chunks(source: str) -> List[str]:
    """ 
    Extract chunks from either a file path or a URL.
    """
    if is_valid_url(source):
        return extract_chunks_from_url(source)
    else:
        # Check file extension and use appropriate function
        if source.lower().endswith('.docx'):
            return extract_chunks_from_docx(source)
        elif source.lower().endswith(('.txt', '.pdf')):
            return extract_chunks_from_other_formats(source)
        else:
            raise ValueError(f"Unsupported file format: {source}")

def is_valid_url(url: str) -> bool:
    """ 
    Check if the given string is a valid URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_chunks_from_url(url: str) -> List[str]:
    """ 
    Extract and dynamically chunk text from a web URL.
    """
    text_content = scrape_web_content(url)
    cleaned_text = clean_paragraph(text_content)
    text_content = cleaned_text if cleaned_text else ""

    preprocessed_text = preprocess_text(text_content)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words from URL after preprocessing: {total_words}")

    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
    logger.info(f"Dynamic chunk size set to {chunk_size} words")

    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks from URL")
    return chunks

def scrape_web_content(url: str) -> str:
    """ 
    Scrape text content from a web page.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text content
            text = soup.get_text(separator="\n", strip=True)
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return text
        else:
            logger.error(f"Failed to fetch page, status code: {response.status_code}")
            raise Exception(f"Failed to fetch web page: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error scraping web content: {str(e)}")
        raise

def clean_paragraph(text: str) -> str:
    """ 
    Clean and normalize paragraph text.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None

def preprocess_text(text: str) -> str:
    """ 
    Preprocess text by lowercasing, removing stopwords, and keeping only alphanumeric tokens.
    """
    if not text or not isinstance(text, str):
        return ""
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word.lower() for word in words if word.lower() not in stop_words and word.isalnum()
    ]
    return " ".join(filtered_words)

def extract_chunks_from_docx(filename: str) -> List[str]:
    """ 
    Extract and dynamically chunk text from a Word document.
    """
    validate_file(filename)
    doc = Document(filename)
    all_text = []
    for para in doc.paragraphs:
        cleaned = clean_paragraph(para.text)
        if cleaned:
            all_text.append(cleaned)
    full_text = " ".join(all_text)
    preprocessed_text = preprocess_text(full_text)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words in document after preprocessing: {total_words}")

    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
    logger.info(f"Dynamic chunk size set to {chunk_size} words")

    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def validate_file(filename: str) -> None:
    """ 
    Validate that the file exists and is a supported format.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    supported_formats = ('.docx', '.txt', '.pdf')
    if not filename.lower().endswith(supported_formats):
        raise ValueError(f"File must be one of {supported_formats}: {filename}")

def extract_chunks_from_other_formats(filename: str) -> List[str]:
    """
    Extract and dynamically chunk text from .txt or .pdf files.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'txt':
        # Read text file
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    elif file_ext == 'pdf':
        # Extract text from PDF using PyPDF2
        text_content = ""
        try:
            with open(filename, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned = clean_paragraph(page_text)
                        if cleaned:
                            text_content += cleaned + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {str(e)}")
            # Fallback to PyMuPDF if available
            try:
                doc = fitz.open(filename)
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        cleaned = clean_paragraph(page_text)
                        if cleaned:
                            text_content += cleaned + "\n"
                doc.close()
            except Exception as e2:
                logger.error(f"Error reading PDF with PyMuPDF: {str(e2)}")
                raise Exception(f"Could not extract text from PDF: {str(e)}, {str(e2)}")
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    # Clean paragraph for .txt as well
    if file_ext == 'txt':
        cleaned_text = clean_paragraph(text_content)
        text_content = cleaned_text if cleaned_text else ""

    # Use the same preprocessing and chunking logic as existing functions
    preprocessed_text = preprocess_text(text_content)
    words = preprocessed_text.split()
    total_words = len(words)
    logger.info(f"Total words from {file_ext.upper()} after preprocessing: {total_words}")
    
    if total_words <= 500:
        chunk_size = 100
    elif total_words <= 2000:
        chunk_size = 250
    else:
        chunk_size = 500
        
    logger.info(f"Dynamic chunk size set to {chunk_size} words")
    
    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    logger.info(f"Created {len(chunks)} chunks from {file_ext.upper()}")
    return chunks

def save_chunks_to_file(chunks: List[str], output_file: str = "chunks.pkl") -> None:
    """ 
    Save extracted chunks to a pickle file.
    """
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")

#   -------------------------------- Phase 2 — Embedding & Indexing ----------------------------------

def initialize_qa_system() -> Tuple[SentenceTransformer, faiss.Index, List[str]]:
    """ 
    Initialize the QA system by loading chunks and creating embeddings.
    """
    print("Loading and embedding chunks...")
    chunks = load_chunks_from_file()
    model, index, embeddings = embed_and_store(chunks)
    print(f"Stored embeddings for {len(chunks)} chunks in FAISS")
    return model, index, chunks

def initialize_qa_system_from_session(session_id: str) -> Tuple[SentenceTransformer, faiss.Index, List[str]]:
    """Initialize QA system from a specific session"""
    print(f"Loading session {session_id}...")
    
    # Get chunks from database
    chunks = get_session_chunks(session_id)
    if not chunks:
        raise ValueError(f"No chunks found for session {session_id}")
    
    # Create embeddings and index
    model, embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    
    print(f"Loaded {len(chunks)} chunks from session {session_id}")
    return model, index, chunks

def load_chunks_from_file(input_file: str = "chunks.pkl") -> List[str]:
    """ 
    Load chunks from a pickle file.
    """
    with open(input_file, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks

def embed_and_store(chunks: List[str]) -> Tuple[SentenceTransformer, faiss.Index, np.ndarray]:
    """ 
    Complete pipeline: embed chunks and store them in a FAISS index.
    """
    model, embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return model, index, embeddings

def create_embeddings(chunks: List[str]) -> Tuple[SentenceTransformer, np.ndarray]:
    """ 
    Create embeddings for text chunks using SentenceTransformer.
    """
    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        embeddings = model.encode(
            chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        ).astype("float32")
        logger.info(f"Successfully embedded {len(chunks)} chunks with dimension {embeddings.shape[1]}")
        return model, embeddings
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """ 
    Create a FAISS index from embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Created FAISS index with dimension {dim}")
    return index

#   ------------------------------ Phase 3 — Retrieval & Answer Generation ---------------------------

def answer_question(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], session_id: str = None) -> str:
    """ 
    Complete QA pipeline for a single question with session tracking.
    """
    # Retrieve relevant chunks
    retrieved = retrieve_relevant_chunks(query, model, index, chunks)
    print(f"\nRetrieved {len(retrieved)} relevant chunks:")
    
    for i, (chunk, score) in enumerate(retrieved, 1):
        # Show FULL chunk without truncation
        print(f"{i}. [Score: {score:.3f}] {chunk}")
        print("---")  # Separator between chunks
    
    # Save user message to database if session_id provided
    if session_id:
        save_chat_message(session_id, 'user', query)
    
    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved)
    enhanced_answer = enhance_answer_quality(answer, query)
    
    # Save bot response to database if session_id provided
    if session_id:
        retrieved_data = [{"text": chunk, "score": float(score)} for chunk, score in retrieved]
        save_chat_message(session_id, 'bot', enhanced_answer, retrieved_data)
    
    return enhanced_answer

def retrieve_relevant_chunks(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
    """ 
    Retrieve the most relevant chunks for a query with their similarity scores.
    Returns FULL chunks without truncation.
    """
    try:
        preprocessed_query = preprocess_text(query)
        query_vec = model.encode([preprocessed_query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(query_vec, top_k)
        
        retrieved = []
        for idx, score in zip(I[0], D[0]):
            if score > 0.15:
                retrieved.append((chunks[idx], float(score)))  # Full chunk, no truncation
        
        retrieved.sort(key=lambda x: x[1], reverse=True)
        
        if not retrieved and I.size > 0:
            for i in range(min(3, len(I[0]))):
                retrieved.append((chunks[I[0][i]], float(D[0][i])))  # Full chunk, no truncation
        
        return retrieved[:top_k]
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunks: {str(e)}")
        return []
    
def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """ 
    Generate an answer using the Gemini API based on retrieved context.
    """
    prompt = construct_enhanced_prompt(query, retrieved_chunks)
    return call_gemini_api(prompt)
 
def construct_enhanced_prompt(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """ 
    Construct a prompt for the LLM that allows mixed formatting:
    - Paragraphs (50–120 words each)
    - Points only when the query explicitly requires it
    """
    if not retrieved_chunks:
        return f"""Answer the following question based on your general knowledge.
Question: {query}

FORMAT RULES:
- Use **bold headings** only when introducing major sections.
- Write in **paragraphs of 30–300 words** when possible and try .
- If the query asks for steps, types, advantages, methods, differences, or lists → use **bullet/numbered points**.
- Otherwise, prefer paragraphs.
- if word 'summarize' come then pls summarize according which give me best summarization.
- Ensure smooth joining of all provided context chunks.
"""

    # Merge retrieved chunks into context
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks):
        clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
        context_parts.append(f"[Source {i+1}, Relevance: {score:.3f}] {clean_chunk}")
    context = "\n\n".join(context_parts)

    # Detect if list structure is needed
    list_indicators = [
        "list", "steps", "ways", "methods", "types", "advantages", "disadvantages", 
        "benefits", "features", "points", "factors", "arguments", "grounds", "reasons", "differences"
    ]
    requires_list_format = any(word in query.lower() for word in list_indicators)

    prompt = f"""You are an expert assistant. Use the retrieved chunks to answer the question.

CRITICAL INSTRUCTIONS:
- Use retrieved chunks directly and merge them smoothly into the answer.
- If chunks are fragmented, rephrase and join them logically.
- Maintain factual accuracy.

FORMAT RULES:
- Use **bold headings** only for section titles (not every line).
- Paragraphs should be in 10 – 300 words depend upon the chunks provied to you so pls see all the chunks provide to you then see the query then genrate the ans and then see it should below 120 words .
- If the question requires a list (e.g., types, steps, advantages), use bullet/numbered points.
- Otherwise, write in paragraphs.
- Highlight key terms in **bold**.

CONTEXT (with relevance scores):
{context}

QUESTION: {query}

Now provide the final answer following the formatting rules.
"""

    if requires_list_format:
        prompt += "\nSince the query expects multiple points, use a list with explanations."
    else:
        prompt += "\nSince the query is descriptive, prefer paragraphs."

    return prompt

def call_gemini_api(prompt: str) -> str:
    """ 
    Make API call to Gemini LLM.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}],
                "role": "user"
            }
        ],
        "generationConfig": {
            "temperature": 0.2,  # Lower temperature for more focused responses
            "topK": 20,
            "topP": 0.8,
            "maxOutputTokens": 4096,  # Increased from 1024 to allow longer responses
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        output = response.json()
        try:
            answer = output["candidates"][0]["content"]["parts"][0]["text"].strip()
            if not answer or len(answer) < 10:
                return "I couldn't generate a satisfactory answer based on the available information. Please try rephrasing your question."
            return answer
        except (KeyError, IndexError):
            logger.warning("Unexpected response format from Gemini API")
            return "I received an unexpected response format from the AI service. Please try again."
    except requests.exceptions.Timeout:
        logger.error("Request to Gemini API timed out")
        return "The AI service is taking too long to respond. Please try again later."
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Gemini API failed: {str(e)}")
        return f"Sorry, I encountered an error connecting to the AI service: {str(e)}"

def enhance_answer_quality(answer: str, query: str) -> str:
    """ 
    Post-process the generated answer to ensure formatting consistency:
    - Paragraphs of ~50–120 words
    - Clean bold headings
    - Lists only when needed
    """
    # Remove boilerplate phrases
    patterns_to_remove = [
        r"Based on the provided context,?",
        r"According to the context,?",
        r"As mentioned in the sources,?",
        r"The context (indicates|shows|states|says) that",
        r"Based on my analysis",
        r"According to the information provided"
    ]
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)

    # Normalize spacing
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'\n\s*\n', '\n\n', answer)

    # Ensure headings are bold
    answer = re.sub(r'(^|\n)([A-Z][A-Za-z ]+:)', r'\n**\2**', answer)

    # Split long paragraphs (>120 words)
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    new_paragraphs, current = [], []
    count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if count + word_count > 120 and current:
            new_paragraphs.append(" ".join(current))
            current = [sentence]
            count = word_count
        else:
            current.append(sentence)
            count += word_count
    if current:
        new_paragraphs.append(" ".join(current))

    return "\n\n".join(new_paragraphs)

# ----------------------------------------------------------------------------------------------------- 

def main():
    """Enhanced main function with session management"""
    try:
        import sys
        session_id = None
        
        # Check for session argument
        if len(sys.argv) > 1 and sys.argv[1] == "--session":
            if len(sys.argv) > 2:
                session_id = sys.argv[2]
                print(f"Resuming session: {session_id}")
                
                # Load existing session
                model, index, chunks = initialize_qa_system_from_session(session_id)
            else:
                # List available sessions
                sessions = get_chat_sessions()
                if sessions:
                    print("\nAvailable chat sessions:")
                    for i, session in enumerate(sessions):
                        print(f"{i+1}. {session['document_source']} ({session['chunk_count']} chunks) - {session['updated_at']}")
                    
                    choice = input("\nEnter session number to resume or 'n' for new session: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                        session_id = sessions[int(choice)-1]['session_id']
                        model, index, chunks = initialize_qa_system_from_session(session_id)
                    else:
                        session_id = None
                else:
                    print("No previous sessions found.")
                    session_id = None
        
        # Start new session if no session loaded
        if not session_id:
            if not os.path.exists("chunks.pkl"):
                source = input("Enter the path to your .docx/.txt/.pdf file or a URL: ").strip()
                if not source:
                    source = "sample.docx"
                chunks, session_id = process_source(source)
            else:
                chunks = load_chunks_from_file()
                session_id = create_chat_session("Existing chunks.pkl", chunks)
            
            # Initialize the QA system
            model, index, chunks = initialize_qa_system()

        print(f"\nSession ID: {session_id}")
        print("Start chatting with the system! (type 'stop chat' to exit)")
        print("Type 'list sessions' to see previous chats or 'switch session' to change session\n")
        
        while True:
            query = input("Enter your query: ").strip()
            
            if query.lower() in ['stop chat', 'exit', 'quit']:
                print("Chat ended.")
                break
            elif query.lower() == 'list sessions':
                sessions = get_chat_sessions()
                if sessions:
                    print("\nPrevious chat sessions:")
                    for i, session in enumerate(sessions):
                        current_indicator = " (CURRENT)" if session['session_id'] == session_id else ""
                        print(f"{i+1}. {session['document_source']} - {session['updated_at']}{current_indicator}")
                else:
                    print("No previous sessions found.")
                continue
            elif query.lower() == 'switch session':
                sessions = get_chat_sessions()
                if sessions:
                    print("\nAvailable sessions:")
                    for i, session in enumerate(sessions):
                        current_indicator = " (CURRENT)" if session['session_id'] == session_id else ""
                        print(f"{i+1}. {session['document_source']} - {session['updated_at']}{current_indicator}")
                    
                    choice = input("\nEnter session number to switch to: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                        new_session_id = sessions[int(choice)-1]['session_id']
                        if new_session_id != session_id:
                            print(f"Switching to session {new_session_id}...")
                            # Restart with new session
                            os.execv(sys.executable, [sys.executable] + sys.argv + ['--session', new_session_id])
                    else:
                        print("Invalid choice.")
                else:
                    print("No sessions available.")
                continue
                
            if not query:
                print("Please enter a valid query.")
                continue

            # Process the query
            answer = answer_question(query, model, index, chunks, session_id)

            # Display the answer
            print("\nFinal Answer:\n")
            print(answer)
            print("-" * 60)

    except FileNotFoundError:
        logger.error("Document file or chunks.pkl not found")
        print("Error: File not found. Please check the file path.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
