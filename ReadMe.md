# Free RAG System - Setup & Run Guide

🚀 A completely **free**, **local** RAG (Retrieval-Augmented Generation) system that runs entirely on your computer with no API costs! Perfect for querying your software documentation using AI.

## 🎯 Quick Start (5 Minutes)

### 1. Install Python Requirements
```bash
pip install sentence-transformers faiss-cpu transformers torch requests numpy
```

### 2. Create Documentation Folder
```bash
mkdir docs
```

### 3. Add Sample Documents
Create these files in the `docs/` folder (see examples below)

### 4. Run the Application
```bash
python rag_system.py
```

That's it! The system will start and you can begin asking questions.

---

## 📋 Detailed Setup Instructions

### Prerequisites
- **Python 3.8+** (Check with `python --version`)
- **4GB RAM minimum** (for model loading)
- **2GB free disk space** (for model downloads)
- **Internet connection** (initial setup only)

### Step 1: Clone/Download the Project
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### Step 2: Install Dependencies

**Option A: Using pip (Recommended)**
```bash
pip install sentence-transformers faiss-cpu transformers torch requests numpy pathlib
```

**Option B: Using requirements.txt (if provided)**
```bash
pip install -r requirements.txt
```

**Troubleshooting Dependencies:**
- If `faiss-cpu` fails: Try `pip install faiss-cpu --no-cache-dir`
- If `torch` is slow: Visit [pytorch.org](https://pytorch.org) for platform-specific install
- For GPU support: Replace `faiss-cpu` with `faiss-gpu`

### Step 3: Set Up Documentation Files

Create a `docs/` folder in your project directory and add your documentation files.

**Required file structure:**
```
your-project/
├── rag_system.py
├── docs/                    # Create this folder
│   ├── api_guide.md        # Your API documentation
│   ├── installation.txt    # Installation instructions
│   └── config.json         # Configuration documentation
└── README.md
```

**Example Documentation Files:**

**`docs/api_guide.md`:**
```markdown
# CloudSync API Guide

## Authentication
To authenticate with the CloudSync API, include your API key in the Authorization header:

```
Authorization: Bearer your-api-key-here
```

## Available Endpoints

### Users
- `GET /api/users` - List all users
- `POST /api/users` - Create new user
- `PUT /api/users/{id}` - Update user

### Synchronization  
- `POST /api/sync` - Start synchronization process
- `GET /api/sync/status` - Check sync status
- `DELETE /api/sync/{id}` - Cancel sync job

## Rate Limits
- 1000 requests per hour per API key
- 10 concurrent connections maximum

## Error Codes
- 401: Unauthorized - Invalid API key
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Contact support
```

**`docs/installation.txt`:**
```
CloudSync Installation Guide

SYSTEM REQUIREMENTS:
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- PostgreSQL 12+ database
- Redis server (for caching)

REQUIRED PORTS:
- Port 8080: Main application
- Port 5432: PostgreSQL database  
- Port 6379: Redis cache
- Port 443: HTTPS traffic

INSTALLATION STEPS:

1. Install Python dependencies:
   pip install -r requirements.txt

2. Configure database:
   - Create database: cloudsync_production
   - Update connection string in config.yaml
   - Run migrations: python manage.py migrate

3. Configure Redis:
   - Install Redis server
   - Update Redis URL in config.yaml
   - Test connection: redis-cli ping

4. Start the application:
   python app.py --port 8080

5. Verify installation:
   - Visit: http://localhost:8080/health
   - Check logs: tail -f logs/application.log

FIREWALL CONFIGURATION:
- Allow inbound: 8080, 443
- Allow outbound: 80, 443, 5432, 6379
- Block all other ports

TROUBLESHOOTING:
- Database connection fails: Check PostgreSQL service status
- Redis errors: Verify Redis server is running
- Port conflicts: Use netstat -an to check port usage
- Permission errors: Run as administrator/sudo
```

**`docs/config.json`:**
```json
{
  "configuration_guide": {
    "database": {
      "connection_string": "postgresql://user:pass@localhost:5432/cloudsync_db",
      "pool_size": 20,
      "timeout": 30,
      "ssl_mode": "require"
    },
    "redis": {
      "url": "redis://localhost:6379",
      "max_connections": 10,
      "timeout": 5
    },
    "sync_settings": {
      "real_time_sync": true,
      "batch_size": 100,
      "retry_attempts": 3,
      "sync_interval": 300
    },
    "security": {
      "api_key_required": true,
      "rate_limit_per_hour": 1000,
      "max_file_size": "10MB",
      "allowed_file_types": ["pdf", "docx", "txt", "md"]
    },
    "logging": {
      "level": "INFO",
      "file_path": "logs/application.log",
      "max_size": "100MB",
      "backup_count": 5
    }
  }
}
```

---

## 🚀 Running the Application

### Basic Run
```bash
python rag_system.py
```

**What happens when you run:**
1. System initializes (downloads models on first run - be patient!)
2. Checks for documentation files in `docs/` folder
3. Processes and creates embeddings for your documents
4. Starts interactive Q&A session

### First Run (Takes 2-5 minutes)
```
🚀 FREE RAG SYSTEM FOR SOFTWARE DOCUMENTATION
============================================================
This system runs entirely on your computer - no API costs!

🔧 Initializing RAG system...
(First run will download models - please be patient)
📥 Loading embedding model: all-MiniLM-L6-v2
📥 Loading Hugging Face model (this may take a few minutes on first run)...
✅ Hugging Face model loaded successfully!

📂 Checking for documentation files...
✅ Found: docs/api_guide.md
✅ Found: docs/installation.txt
✅ Found: docs/config.json

📚 Loading and processing 3 documents...
🔄 Creating embeddings for 12 documents...
✅ Added 12 documents to vector store
✅ Successfully loaded and processed all documents!

💾 Saving RAG system...
✅ RAG system saved!

============================================================
🎉 RAG SYSTEM READY!
Ask questions about your CloudSync documentation.
Type 'quit' to exit
============================================================
```

### Subsequent Runs (Takes 10-30 seconds)
After the first run, the system starts much faster as models are cached.

---

## 💬 Using the System

### Sample Questions to Try:
```
🤔 Your question: How do I authenticate with the API?
🤔 Your question: What are the system requirements for installation?  
🤔 Your question: How do I configure the database connection?
🤔 Your question: What endpoints are available in the API?
🤔 Your question: What firewall ports need to be open?
🤔 Your question: How do I enable real-time sync?
```

### Example Interaction:
```
🤔 Your question: How do I authenticate with the API?

🤖 Thinking... (this may take 10-30 seconds)

✅ Answer:
To authenticate with the CloudSync API, you need to include your API key in the Authorization header using the Bearer token format: "Authorization: Bearer your-api-key-here". The API requires authentication for all endpoints and has rate limits of 1000 requests per hour per API key.

📚 Sources:
📄 api_guide.md
------------------------------------------------------------
```

### Commands:
- **Ask any question** about your documentation
- **`quit`** or **`exit`** or **`q`** - Exit the program
- **Ctrl+C** - Force quit

---

## ⚙️ Configuration Options

### Choose Your AI Backend

**Option 1: Hugging Face (Default - Simpler Setup)**
```python
# In rag_system.py, line ~154:
rag = FreeRAGSystem(llm_backend="huggingface", always_rebuild=True)
```

**Option 2: Ollama (Better Responses)**
```python
# In rag_system.py, line ~154:
rag = FreeRAGSystem(llm_backend="ollama", always_rebuild=True)
```

To use Ollama:
1. Install Ollama: [ollama.ai](https://ollama.ai)
2. Install a model: `ollama pull llama2`
3. Start Ollama: `ollama serve`
4. Change the backend in the code

### Customize Document Processing
```python
# Adjust chunk size and overlap
processor = DocumentProcessor(
    chunk_size=512,    # Characters per chunk
    chunk_overlap=50   # Overlap between chunks
)
```

### Performance Tuning
**For Speed:**
- Use `chunk_size=256`
- Set `always_rebuild=False` after first run
- Use fewer retrieval results: `k=3`

**For Quality:**
- Use `chunk_size=1024`
- Use more retrieval results: `k=5`
- Use Ollama backend with larger models

---

## 🔧 Troubleshooting

### Common Issues

**❌ "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**❌ "faiss-cpu not found"**
```bash
pip install faiss-cpu --no-cache-dir
```

**❌ "No documents loaded"**
- Check that `docs/` folder exists
- Verify files exist: `docs/api_guide.md`, `docs/installation.txt`, `docs/config.json`
- Check file permissions

**❌ "Error initializing RAG system"**
```bash
# Install all dependencies
pip install sentence-transformers faiss-cpu transformers torch requests numpy

# If still failing, try one by one:
pip install sentence-transformers
pip install faiss-cpu  
pip install transformers
pip install torch
```

**❌ "Model download is slow"**
- First run downloads ~400MB of models
- Be patient (2-5 minutes)
- Models are cached for future runs

**❌ "Out of memory"**
- Close other applications
- Use smaller chunk_size (256)
- Reduce number of documents

**❌ "Ollama connection failed"**
```bash
# Make sure Ollama is installed and running
ollama serve

# Make sure you have a model
ollama pull llama2

# Test connection
curl http://localhost:11434/api/version
```

### Debug Mode
Add print statements to see what's happening:
```python
# Check document loading
print(f"Looking for files: {doc_paths}")
for path in doc_paths:
    print(f"File exists: {Path(path).exists()} - {path}")
```

### Check System Status
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(sentence|faiss|transformers|torch)"

# Check available memory
# Windows: wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
# Linux/Mac: free -h
```

---

## 📁 File Structure After Setup

```
your-project/
├── rag_system.py              # Main application
├── README.md                  # This file
├── .gitignore                 # Excludes large files
├── requirements.txt           # Dependencies
├── docs/                      # Your documentation
│   ├── api_guide.md
│   ├── installation.txt  
│   └── config.json
├── rag_index/                 # Generated (auto-created)
│   ├── index.faiss           # Vector embeddings
│   └── documents.json        # Document metadata
└── .cache/                    # Model cache (auto-created)
    └── huggingface/
        └── (model files)
```

---

## 🚀 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install sentence-transformers faiss-cpu transformers torch requests numpy`
- [ ] Created `docs/` folder
- [ ] Added documentation files (api_guide.md, installation.txt, config.json)
- [ ] Run: `python rag_system.py`
- [ ] Wait for models to download (first run only)
- [ ] Start asking questions!

---

## 💡 Tips for Best Results

1. **Write clear questions**: "How do I configure the database?" vs "database stuff"
2. **Use specific terms**: Use exact terminology from your docs
3. **Ask follow-up questions**: Build on previous answers
4. **Check sources**: The system shows which documents it used
5. **Update docs regularly**: The system rebuilds embeddings to catch changes

---

**🎉 You're ready to go! Run `python rag_system.py` and start querying your documentation with AI!**
