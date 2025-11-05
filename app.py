from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename

# Import your embedding.py functionss
import embedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store the QA system state
model = None
index = None
chunks = None
current_session_id = None
is_initialized = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index_page():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        chunk_count = len(chunks) if chunks else 0
        return jsonify({
            'status': 'success',
            'initialized': is_initialized,
            'chunk_count': chunk_count,
            'session_id': current_session_id,
            'chunks_file_exists': os.path.exists('chunks.pkl')
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    try:
        sessions = embedding.get_chat_sessions()
        return jsonify({
            'status': 'success',
            'sessions': sessions,
            'current_session': current_session_id
        })
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get chat history for a specific session"""
    try:
        history = embedding.get_chat_history(session_id)
        return jsonify({
            'status': 'success',
            'history': history,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    try:
        success = embedding.delete_chat_session(session_id)
        if success:
            # If current session is deleted, reset system
            global current_session_id, is_initialized, model, index, chunks
            if session_id == current_session_id:
                model = None
                index = None
                chunks = None
                current_session_id = None
                is_initialized = False
            return jsonify({'status': 'success', 'message': 'Session deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete session'}), 500
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/process-file', methods=['POST'])
def process_file():
    """Process uploaded file"""
    global is_initialized, current_session_id
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error', 
                'message': 'File type not allowed. Use .txt, .pdf, or .docx'
            }), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filepath}")
        
        # Process the file using your embedding.py function
        processed_chunks, session_id = embedding.process_source(filepath)
        current_session_id = session_id
        
        # Reset initialization flag
        is_initialized = False
        
        return jsonify({
            'status': 'success',
            'message': f'File processed! Created {len(processed_chunks)} chunks.',
            'chunk_count': len(processed_chunks),
            'filename': filename,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/process-url', methods=['POST'])
def process_url():
    """Process URL"""
    global is_initialized, current_session_id
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400
        
        logger.info(f"Processing URL: {url}")
        
        # Process the URL using your embedding.py function
        processed_chunks, session_id = embedding.process_source(url)
        current_session_id = session_id
        
        # Reset initialization flag
        is_initialized = False
        
        return jsonify({
            'status': 'success',
            'message': f'URL processed! Created {len(processed_chunks)} chunks.',
            'chunk_count': len(processed_chunks),
            'url': url,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the QA system"""
    global model, index, chunks, is_initialized
    
    try:
        # Check if chunks.pkl exists
        if not os.path.exists('chunks.pkl'):
            return jsonify({
                'status': 'error',
                'message': 'Please process a document first!'
            }), 400
        
        logger.info("Initializing QA system...")
        
        # Initialize using your embedding.py function
        model, index, chunks = embedding.initialize_qa_system()
        is_initialized = True
        
        logger.info(f"System initialized with {len(chunks)} chunks")
        
        return jsonify({
            'status': 'success',
            'message': f'System ready with {len(chunks)} chunks!',
            'chunk_count': len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        is_initialized = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/load', methods=['POST'])
def load_session(session_id):
    """Load a specific chat session"""
    global model, index, chunks, current_session_id, is_initialized
    
    try:
        logger.info(f"Loading session: {session_id}")
        
        # Load session from database
        model, index, chunks = embedding.initialize_qa_system_from_session(session_id)
        current_session_id = session_id
        is_initialized = True
        
        # Get chat history
        history = embedding.get_chat_history(session_id)
        
        logger.info(f"Session {session_id} loaded with {len(chunks)} chunks and {len(history)} messages")
        
        return jsonify({
            'status': 'success',
            'message': f'Session loaded!',
            'chunk_count': len(chunks),
            'session_id': session_id,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_system():
    """Answer a question"""
    global model, index, chunks, current_session_id, is_initialized
    
    try:
        # Check if system is initialized
        if not is_initialized or model is None or index is None or chunks is None:
            return jsonify({
                'status': 'error',
                'message': 'System not initialized. Please process a document first!'
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'status': 'error', 'message': 'No question provided'}), 400
        
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant chunks
        retrieved_chunks = embedding.retrieve_relevant_chunks(question, model, index, chunks)
        
        # Format chunks for response
        chunks_data = [
            {
                'text': chunk_text,
                'score': float(score)
            }
            for chunk_text, score in retrieved_chunks
        ]
        
        # Generate answer using your embedding.py function
        answer = embedding.answer_question(question, model, index, chunks, current_session_id)
        
        logger.info("Answer generated successfully")
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'retrieved_chunks': chunks_data,
            'chunk_count': len(retrieved_chunks),
            'session_id': current_session_id
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the system and clear all data"""
    global model, index, chunks, current_session_id, is_initialized
    
    try:
        model = None
        index = None
        chunks = None
        current_session_id = None
        is_initialized = False
        
        # Delete chunks.pkl if exists
        if os.path.exists('chunks.pkl'):
            os.remove('chunks.pkl')
        
        # Clear uploads folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        logger.info("System reset successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'System reset successfully!'
        })
        
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RAG Document Q&A System - Flask Backend")
    print("="*60)
    print("\n📝 Setup Instructions:")
    print("1. Ensure 'embedding.py' is in the same folder")
    print("2. Create a 'static' folder and place the HTML file inside")
    print("3. Make sure MySQL database is running")
    print("4. Update DB_CONFIG in embedding.py with your credentials")
    print("5. Server starting at: http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
