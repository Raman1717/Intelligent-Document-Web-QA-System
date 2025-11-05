import re 
import os 
import pickle 
import requests 
import json 
import faiss 
import numpy as np 
import logging 
from docx import Document 
from typing import List, Tuple, Optional, Union 
from sentence_transformers import SentenceTransformer 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from bs4 import BeautifulSoup 
from urllib.parse import urlparse 
import PyPDF2
import fitz  # PyMuPDF
from openai import OpenAI

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
HF_API_KEY = "............................."
HF_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp:novita"
HF_MODEL = "openai/gpt-oss-20b:fireworks-ai"
DEFAULT_TOP_K = 3

# Initialize OpenAI client for HuggingFace
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

# ------------------ TEXT PREPROCESSING FUNCTIONS ------------------ 
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

# ------------------ DOCUMENT PROCESSING FUNCTIONS ------------------ 
def clean_paragraph(text: str) -> str:
    """ 
    Clean and normalize paragraph text.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None

def is_valid_url(url: str) -> bool:
    """ 
    Check if the given string is a valid URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

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

def save_chunks_to_file(chunks: List[str], output_file: str = "chunks.pkl") -> None:
    """ 
    Save extracted chunks to a pickle file.
    """
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")

def load_chunks_from_file(input_file: str = "chunks.pkl") -> List[str]:
    """ 
    Load chunks from a pickle file.
    """
    with open(input_file, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks

# ------------------ EMBEDDING & STORAGE FUNCTIONS ------------------ 
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

def embed_and_store(chunks: List[str]) -> Tuple[SentenceTransformer, faiss.Index, np.ndarray]:
    """ 
    Complete pipeline: embed chunks and store them in a FAISS index.
    """
    model, embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return model, index, embeddings

# ------------------ RETRIEVAL FUNCTIONS ------------------ 
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

# ------------------ PROMPT ENGINEERING FUNCTIONS ------------------ 
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
- Only use the information provided in the given chunks to generate answers. Do not add, infer, or use any external knowledge—if something is missing, state “Not mentioned in the provided context.”
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

# ------------------ LLM INTERACTION FUNCTIONS ------------------ 
def call_huggingface_api(prompt: str) -> str:
    """ 
    Make API call to HuggingFace OpenAI-compatible API.
    """
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            top_p=0.8,
            max_tokens=4096,
        )
        answer = completion.choices[0].message.content.strip()
        if not answer or len(answer) < 10:
            return "I couldn't generate a satisfactory answer based on the available information. Please try rephrasing your question."
        return answer
    except Exception as e:
        logger.error(f"Request to HuggingFace API failed: {str(e)}")
        return f"Sorry, I encountered an error connecting to the AI service: {str(e)}"

def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """ 
    Generate an answer using the HuggingFace API based on retrieved context.
    """
    prompt = construct_enhanced_prompt(query, retrieved_chunks)
    return call_huggingface_api(prompt)

# ------------------ ANSWER PROCESSING FUNCTIONS ------------------ 
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

# ------------------ MAIN APPLICATION FUNCTIONS ------------------ 
def process_source(source: str) -> List[str]:
    """ 
    Complete source processing pipeline for either file or URL.
    """

    print("📄 Processing document and URL ...")
    chunks = extract_chunks(source)  # This will handle all file formats now
    save_chunks_to_file(chunks)
    print(f"✅ Extracted {len(chunks)} chunks and saved to chunks.pkl")
    return chunks

def initialize_qa_system() -> Tuple[SentenceTransformer, faiss.Index, List[str]]:
    """ 
    Initialize the QA system by loading chunks and creating embeddings.
    """
    print("🔄 Loading and embedding chunks...")
    chunks = load_chunks_from_file()
    model, index, embeddings = embed_and_store(chunks)
    print(f"✅ Stored embeddings for {len(chunks)} chunks in FAISS")
    return model, index, chunks

def answer_question(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str]) -> str:
    """ 
    Complete QA pipeline for a single question.
    """
    # Retrieve relevant chunks
    retrieved = retrieve_relevant_chunks(query, model, index, chunks)
    print(f"\n🔎 Retrieved {len(retrieved)} relevant chunks:")
    
    for i, (chunk, score) in enumerate(retrieved, 1):
        # Show FULL chunk without truncation
        print(f"{i}. [Score: {score:.3f}] {chunk}")
        print("---")  # Separator between chunks
    
    # Generate answer
    print("\n🤖 Generating answer...")
    answer = generate_answer(query, retrieved)
    enhanced_answer = enhance_answer_quality(answer, query)
    
    return enhanced_answer

def main():
    """Main function to run the complete QA system."""
    try:
        # Check if we need to process a source
        if not os.path.exists("chunks.pkl"):
            source = input("Enter the path to your .docx/.txt/.pdf file or a URL: ").strip()
            if not source:
                source = "sample.docx"
            process_source(source)

        # Initialize the QA system
        model, index, chunks = initialize_qa_system()

        # Chat loop
        print("\n💬 Start chatting with the system! (type 'stop chat' to exit)\n")
        while True:
            query = input("👉 Enter your query: ").strip()
            if query.lower() in ['stop chat', 'exit', 'quit']:
                print("👋 Chat ended.")
                break
            if not query:
                print("Please enter a valid query.")
                continue

            # Process the query
            answer = answer_question(query, model, index, chunks)

            # Display the answer
            print("\n📝 Final Answer:\n")
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

