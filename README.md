<h1 align="center">🤖 RAG Document Q&A System</h1>
<p align="center">
  <em>A sophisticated Retrieval-Augmented Generation (RAG) system that allows users to upload documents, process them, and ask questions using AI-powered responses.</em>
</p>

<hr>

<h2>✨ Features</h2>

<h3>🔐 Authentication & Security</h3>
<ul>
  <li>User Registration & Login with secure token-based authentication</li>
  <li>Password strength validation with visual indicators</li>
  <li>Session management with automatic token verification</li>
  <li>Secure password hashing using Werkzeug</li>
</ul>

<h3>📄 Document Processing</h3>
<ul>
  <li><b>Multiple Input Methods:</b> File upload (PDF, DOCX, TXT) or URL processing</li>
  <li><b>OCR Integration:</b> Text extraction from images within documents</li>
  <li><b>Smart Chunking:</b> Dynamic chunk sizing based on document length</li>
  <li><b>Image Content Recognition:</b> Extracts and processes text from images in PDFs, DOCX files, and web pages</li>
</ul>

<h3>💬 Intelligent Q&A</h3>
<ul>
  <li><b>AI-Powered Responses</b> using Google’s Gemini API</li>
  <li><b>Context-Aware Answers</b> based on retrieved document chunks</li>
  <li><b>Smart Formatting:</b> Bullet points, numbered lists, and bold text</li>
  <li><b>Relevance Scoring:</b> Displays confidence levels for retrieved information</li>
</ul>

<h3>🗂️ Session Management</h3>
<ul>
  <li>Multiple Chat Sessions per user</li>
  <li>Automatic session history saving</li>
  <li>Switch easily between different document sessions</li>
  <li>Delete sessions with confirmation prompts</li>
</ul>

<h3>🎨 User Interface</h3>
<ul>
  <li>Modern Dark/Light Theme toggle</li>
  <li>Responsive Design (desktop & mobile)</li>
  <li>Real-time Status Indicators</li>
  <li>Loading States and progress indicators</li>
  <li>Clean, Intuitive Layout with sidebar navigation</li>
</ul>

<hr>

<h2>🛠️ Technology Stack</h2>

<h3>Backend</h3>
<ul>
  <li><b>Flask</b> – Web framework</li>
  <li><b>MySQL</b> – Database storage</li>
  <li><b>Sentence Transformers</b> – Text embeddings</li>
  <li><b>FAISS</b> – Vector similarity search</li>
  <li><b>Google Gemini API</b> – LLM for answer generation</li>
</ul>

<h3>Frontend</h3>
<ul>
  <li>HTML5 / CSS3 – Modern responsive design</li>
  <li>JavaScript – No framework dependencies</li>
  <li>CSS Grid & Flexbox – Layout management</li>
</ul>

<h3>Processing Libraries</h3>
<ul>
  <li>PyMuPDF – PDF text extraction</li>
  <li>python-docx – DOCX processing</li>
  <li>BeautifulSoup4 – Web scraping</li>
  <li>Pytesseract – OCR for images</li>
  <li>NLTK – Text preprocessing</li>
</ul>

<hr>

<h2>📋 Prerequisites</h2>
<ul>
  <li>Python 3.8+</li>
  <li>MySQL Server 8.0+</li>
  <li>Tesseract OCR (for image text extraction)</li>
  <li>Google Gemini API Key</li>
</ul>

<hr>

<h2>🚀 Installation</h2>

<ol>
  <li><b>Database Setup</b><br>
    Create the MySQL database and tables using the provided SQL schema:
    <ul>
      <li>Users table for authentication</li>
      <li>Chat sessions table for session management</li>
      <li>Chat messages table for conversation history</li>
      <li>Session chunks table for document storage</li>
    </ul>
  </li>

  <li><b>Python Dependencies</b><br>
    Install required packages:
    <ul>
      <li>Flask, Flask-CORS</li>
      <li>mysql-connector-python</li>
      <li>sentence-transformers, faiss-cpu</li>
      <li>nltk, requests, beautifulsoup4</li>
      <li>PyPDF2, PyMuPDF, python-docx</li>
      <li>pytesseract, pillow</li>
    </ul>
  </li>

  <li><b>Configuration</b><br>
    Update MySQL credentials and Google Gemini API key in the app configuration.
  </li>

  <li><b>Tesseract OCR Setup</b><br>
    Install and configure Tesseract OCR for your OS (Windows, macOS, or Linux).
  </li>
</ol>

<hr>

<h2>🎯 Usage</h2>
<ol>
  <li>Start the Flask backend server</li>
  <li>Access the web app in your browser</li>
  <li>Create an account or sign in</li>
  <li>Upload documents via file upload or URL</li>
  <li>Ask questions and receive AI-powered responses!</li>
</ol>

<h3>Document Processing</h3>
<ul>
  <li>PDF, DOCX, TXT file support</li>
  <li>Web page URL processing</li>
  <li>OCR for text extraction from images</li>
  <li>Smart Chunking for optimal performance</li>
</ul>

<h3>Session Management</h3>
<ul>
  <li>Switch between multiple chat sessions</li>
  <li>Each session maintains its own context</li>
  <li>Delete unwanted sessions securely</li>
</ul>

<hr>

<h2>🔧 API Endpoints</h2>

<ul>
  <li><b>Authentication:</b> Register, login, logout, token verification</li>
  <li><b>Document Processing:</b> Upload, URL process, reset</li>
  <li><b>Chat & Sessions:</b> Query, manage sessions, retrieve history, delete</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>
<ul>
  <li>Flask backend for API and logic</li>
  <li>RAG processing module for AI and embeddings</li>
  <li>Frontend with responsive web design</li>
  <li>Upload directory for temporary files</li>
  <li>Cache for processed document chunks</li>
</ul>

<hr>

<h2>⚙️ Configuration Options</h2>
<ul>
  <li><b>Database Settings:</b> Configure MySQL connection parameters</li>
  <li><b>API Configuration:</b> Set Gemini API key and model parameters</li>
  <li><b>Processing Settings:</b> Adjust chunk sizes, retrieval params, OCR options</li>
</ul>

<hr>

<h2>🐛 Troubleshooting</h2>
<ul>
  <li><b>Database Connection Errors:</b> Check MySQL service and credentials</li>
  <li><b>OCR Problems:</b> Verify Tesseract installation and image quality</li>
  <li><b>API Key Issues:</b> Validate Gemini API key and connectivity</li>
  <li><b>File Upload Limits:</b> Adjust maximum file size</li>
</ul>

<hr>

<h2>🔒 Security Features</h2>
<ul>
  <li>Secure password hashing & token authentication</li>
  <li>SQL injection prevention via parameterized queries</li>
  <li>File type validation and secure uploads</li>
  <li>User session isolation and data protection</li>
</ul>

<hr>

<h3 align="center">🚀 Built with ❤️ using Flask, FAISS, and Google Gemini API</h3>
