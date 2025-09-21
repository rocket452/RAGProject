import os
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss

# Option 1: Using Hugging Face Transformers (Free)
from transformers import pipeline

# Option 2: Using Ollama (Free, requires Ollama installation)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

@dataclass
class Document:
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # Initialize the document processor with configurable chunk parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, doc_paths: List[str]) -> List[Document]:
        # Load documents from file paths, supporting .txt, .md, and .json formats
        # Returns a list of Document objects with content split into chunks
        """Load documents from various file formats"""
        documents = []
        
        for path in doc_paths:
            path = Path(path)
            
            if not path.exists():
                print(f"⚠️  Warning: Document not found: {path}")
                continue
                
            if path.suffix.lower() == '.txt':
                content = path.read_text(encoding='utf-8')
            elif path.suffix.lower() == '.md':
                content = path.read_text(encoding='utf-8')
            elif path.suffix.lower() == '.json':
                data = json.loads(path.read_text())
                content = json.dumps(data, indent=2)
            else:
                print(f"Unsupported file type: {path.suffix}")
                continue
            
            # Create chunks
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    metadata={
                        'source': str(path),
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        # Split text into overlapping chunks based on sentences, respecting chunk_size and chunk_overlap
        """Split text into overlapping chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class VectorStore:
    """Handles embeddings and vector similarity search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize the vector store with a sentence transformer model for creating embeddings
        print(f"📥 Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.dimension = None
    
    def add_documents(self, documents: List[Document]):
        # Generate embeddings for documents and add them to the FAISS index for similarity search
        """Add documents and create embeddings"""
        print(f"🔄 Creating embeddings for {len(documents)} documents...")
        
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        self.documents.extend(documents)
        
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
        
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype('float32'))
        
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        # Find the k most similar documents to the query using cosine similarity
        """Search for most relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_normalized.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        # Save the FAISS index and document metadata to disk for later use
        """Save the vector store to disk"""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'embedding': doc.embedding.tolist() if doc.embedding is not None else None
            })
        
        with open(path / "documents.json", 'w') as f:
            json.dump(docs_data, f, indent=2)
    
    def load(self, path: str):
        # Load a previously saved FAISS index and document metadata from disk
        """Load the vector store from disk"""
        path = Path(path)
        
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.dimension = self.index.d
        
        with open(path / "documents.json", 'r') as f:
            docs_data = json.load(f)
        
        self.documents = []
        for doc_data in docs_data:
            doc = Document(
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                embedding=np.array(doc_data['embedding']) if doc_data['embedding'] else None
            )
            self.documents.append(doc)

class FreeRAGSystem:
    """RAG system using free local models"""
    
    def __init__(self, llm_backend: str = "huggingface", always_rebuild: bool = True):
        # Initialize the RAG system with specified language model backend (huggingface, ollama, or template)
        self.vector_store = VectorStore()
        self.processor = DocumentProcessor()
        self.llm_backend = llm_backend
        self.always_rebuild = always_rebuild
        
        if llm_backend == "huggingface":
            print("📥 Loading Hugging Face model (this may take a few minutes on first run)...")
            try:
                # Using a simpler model for better compatibility
                self.generator = pipeline(
                    "text-generation",
                    model="gpt2",  # More reliable than DialoGPT
                    device_map="auto",
                    max_length=512
                )
                print("✅ Hugging Face model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Error loading Hugging Face model: {e}")
                print("📝 Falling back to template-only responses...")
                self.llm_backend = "template"
        elif llm_backend == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ValueError("requests library not available for Ollama backend")
            print("🔗 Using Ollama backend (make sure Ollama is running with a model installed)")
    
    def add_documents(self, doc_paths: List[str], force_rebuild: bool = None):
        # Load documents from file paths and add them to the vector store, optionally rebuilding embeddings
        """Add documents to the knowledge base"""
        rebuild = force_rebuild if force_rebuild is not None else self.always_rebuild
        
        if rebuild:
            print("🔄 Rebuilding document embeddings (fresh conversion)...")
            self.vector_store.documents = []
            self.vector_store.index = None
            self.vector_store.dimension = None
            
        documents = self.processor.load_documents(doc_paths)
        if documents:
            self.vector_store.add_documents(documents)
        else:
            print("❌ No documents loaded! Please check your file paths.")
    
    def generate_with_huggingface(self, prompt: str) -> str:
        # Generate text response using Hugging Face transformers (GPT-2 model)
        """Generate response using Hugging Face transformers"""
        try:
            # Keep prompt shorter for GPT-2
            max_prompt_length = 300
            if len(prompt) > max_prompt_length:
                # Truncate but keep the question
                lines = prompt.split('\n')
                question_line = [line for line in lines if line.startswith('Question:')]
                context_lines = [line for line in lines if not line.startswith('Question:') and line.strip()]
                
                # Keep question and some context
                short_context = '\n'.join(context_lines[:3])  # First 3 context lines
                prompt = f"{short_context}\n\n{question_line[0] if question_line else 'Question: Please answer based on the context.'}\n\nAnswer:"
            
            responses = self.generator(
                prompt,
                max_new_tokens=100,  # Limit response length
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=True
            )
            
            generated = responses[0]['generated_text']
            # Extract only the new part (after the prompt)
            response = generated[len(prompt):].strip()
            
            # Clean up the response
            if response:
                # Take first sentence/paragraph
                response = response.split('\n')[0].strip()
                if len(response) > 200:
                    response = response[:200] + "..."
                return response
            else:
                return "Based on the documentation provided, I can help answer your question about the CloudSync system."
                
        except Exception as e:
            return f"I can help with your question based on the documentation. (Note: {str(e)})"
    
    def generate_with_ollama(self, prompt: str) -> str:
        # Generate text response using Ollama API with a local language model
        """Generate response using Ollama"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def simple_template_response(self, context: str, question: str) -> str:
        # Generate a basic template-based response when language models are unavailable
        """Simple template-based response for when LLMs aren't available"""
        return f"Based on the documentation, here's the relevant information:\n\n{context[:500]}{'...' if len(context) > 500 else ''}\n\nThis should help answer your question about: {question}"
    
    def query(self, question: str, k: int = 3, include_sources: bool = True) -> str:
        # Main RAG function: retrieve relevant documents and generate an answer using the configured LLM
        """Answer a question using RAG"""
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(question, k=k)
        
        if not relevant_docs:
            return "❌ I don't have any relevant information to answer this question. Please check that documents were loaded correctly."
        
        # Prepare context
        context_parts = []
        sources = set()
        
        for doc, score in relevant_docs:
            context_parts.append(doc.content)
            sources.add(doc.metadata['source'])
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following documentation, please answer the question clearly.

Documentation:
{context}

Question: {question}

Answer:"""
        
        # Generate response based on backend
        if self.llm_backend == "huggingface":
            answer = self.generate_with_huggingface(prompt)
        elif self.llm_backend == "ollama":
            answer = self.generate_with_ollama(prompt)
        else:
            # Fallback to simple template
            answer = self.simple_template_response(context, question)
        
        # Add sources if requested
        if include_sources and sources:
            source_list = "\n".join([f"📄 {Path(source).name}" for source in sorted(sources)])
            answer += f"\n\n📚 Sources:\n{source_list}"
        
        return answer
    
    def save(self, path: str):
        # Save the entire RAG system (vector store and embeddings) to disk
        """Save the RAG system"""
        self.vector_store.save(path)
    
    def load(self, path: str):
        # Load a previously saved RAG system from disk
        """Load the RAG system"""
        self.vector_store.load(path)

def main():
    # Main function that sets up and runs the interactive RAG system
    print("=" * 60)
    print("🚀 FREE RAG SYSTEM FOR SOFTWARE DOCUMENTATION")
    print("=" * 60)
    print("This system runs entirely on your computer - no API costs!")
    print("")
    
    # Initialize RAG system with Hugging Face (always rebuild mode)
    print("🔧 Initializing RAG system...")
    print("(First run will download models - please be patient)")
    
    try:
        #rag = FreeRAGSystem(llm_backend="huggingface", always_rebuild=True)
        rag = FreeRAGSystem(llm_backend="ollama", always_rebuild=True)
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("\n🔧 Make sure you have installed all dependencies:")
        print("pip install sentence-transformers faiss-cpu transformers torch")
        return
    
    # Check for documentation files
    doc_paths = [
        "docs/api_guide.md",
        "docs/installation.txt", 
        "docs/config.json"
    ]
    
    print(f"\n📂 Checking for documentation files...")
    missing_files = []
    for path in doc_paths:
        if Path(path).exists():
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files. Please create the docs/ folder and add the files.")
        print("You can find the sample files in the setup guide.")
        return
    
    print(f"\n📚 Loading and processing {len(doc_paths)} documents...")
    print("🔄 This will rebuild embeddings every time (detecting fresh changes)")
    
    try:
        rag.add_documents(doc_paths)
        print(f"✅ Successfully loaded and processed all documents!")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return
    
    # Save the system for potential future use
    print("\n💾 Saving RAG system...")
    try:
        rag.save("rag_index")
        print("✅ RAG system saved!")
    except Exception as e:
        print(f"⚠️  Warning: Could not save system: {e}")
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("🎉 RAG SYSTEM READY!")
    print("Ask questions about your CloudSync documentation.")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    # Sample questions to try
    print("\n💡 Try these sample questions:")
    sample_questions = [
        "How do I authenticate with the API?",
        "What are the system requirements for installation?",
        "How do I configure the database connection?",
        "What endpoints are available in the API?",
        "How do I enable real-time sync?",
        "What firewall ports need to be open?"
    ]
    
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "-" * 60)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
                
            print("\n🤖 Thinking... (this may take 10-30 seconds)")
            answer = rag.query(question)
            print(f"\n✅ Answer:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()
