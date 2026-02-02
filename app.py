import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# ============ LANGSMITH SETUP ============
# Enable tracing but optimize for free tier
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Keep enabled
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "medical-chatbot-render"

# OPTIMIZATION: Sample traces (1 in 10 requests) to save costs
os.environ["LANGSMITH_SAMPLE_RATE"] = "0.1"  # 10% sampling rate

print(f"🔍 LangSmith: Tracing enabled with sampling (10%)")
# =========================================

from flask import Flask, render_template, request, session, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from src.prompt import *
import uuid
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")

# Initialize components
try:
    logger.info("Initializing components...")
    embeddings = download_embeddings()
    
    index_name = "medicalchatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    
    retriever = docsearch.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 2}
    )
    
    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=500
    )
    
    logger.info("✅ All components initialized")
    
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    embeddings = None
    retriever = None
    chat_model = None

# Memory storage
session_memories = {}

def get_or_create_memory(session_id):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    get_or_create_memory(session['session_id'])
    return render_template('chat.html')

@app.route('/get', methods=["POST"])
def chat():
    """Handle chat messages with optimized LangSmith tracing"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return "Session expired. Please refresh the page."
        
        msg = request.form['msg']
        logger.info(f"User ({session_id[:8]}): {msg[:50]}...")
        
        if not all([chat_model, retriever]):
            return "Service is initializing. Please try again in 30 seconds."
        
        memory = get_or_create_memory(session_id)
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # ============ OPTIMIZED TRACING ============
        # Use LangChain's auto-tracing (already enabled via env vars)
        # No need for manual trace() calls - saves overhead
        
        # Create and execute chain (automatically traced by LangSmith)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        question_answering_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answering_chain)
        
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        
        answer = response['answer']
        
        # Save to memory
        memory.save_context({"input": msg}, {"output": answer})
        
        # Log to LangSmith via metadata (optional)
        try:
            from langsmith import trace
            
            # Add custom metadata with low overhead
            trace.get_current_run().metadata.update({
                "session_id": session_id[:8],
                "memory_size": len(chat_history),
                "sampled": random.random() < 0.1  # Mark if sampled
            })
        except:
            pass  # Silently fail if LangSmith not available
        
        logger.info(f"Bot response ({session_id[:8]}): {answer[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        
        # Log error to LangSmith
        try:
            from langsmith import trace
            trace.get_current_run().tags.append("error")
        except:
            pass
        
        return "I'm having trouble processing your request. Please try again."

@app.route('/health')
def health():
    """Health check with LangSmith status"""
    langsmith_status = "disabled"
    try:
        from langsmith import Client
        client = Client()
        langsmith_status = "connected"
    except:
        langsmith_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "langsmith": langsmith_status,
        "components": {
            "pinecone": bool(retriever),
            "groq": bool(chat_model),
            "embeddings": bool(embeddings)
        }
    })

@app.route('/langsmith-test')
def langsmith_test():
    """Test LangSmith connection"""
    try:
        from langsmith import Client
        client = Client()
        
        # Create a test trace
        from langsmith import trace
        
        with trace(name="LangSmith_Test", run_type="chain") as run:
            test_result = "LangSmith is working!"
            
            # Add metadata
            run.metadata = {
                "test": True,
                "environment": "render-production",
                "timestamp": str(uuid.uuid4())
            }
        
        return jsonify({
            "status": "success",
            "message": "LangSmith test trace created",
            "run_id": run.id,
            "langsmith_url": f"https://smith.langchain.com/o/{os.environ.get('LANGCHAIN_PROJECT', 'default')}/r/{run.id}"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "env_vars": {
                "LANGCHAIN_API_KEY_set": bool(os.environ.get("LANGCHAIN_API_KEY")),
                "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT", "not set")
            }
        })

@app.route('/toggle-tracing', methods=['POST'])
def toggle_tracing():
    """Toggle LangSmith tracing on/off (admin endpoint)"""
    current = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    new_value = "false" if current == "true" else "true"
    os.environ["LANGCHAIN_TRACING_V2"] = new_value
    
    return jsonify({
        "status": "tracing_toggled",
        "new_value": new_value,
        "message": f"Tracing {'enabled' if new_value == 'true' else 'disabled'}"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Medical Chatbot starting on port {port}")
    logger.info(f"🔍 LangSmith: {os.environ.get('LANGCHAIN_TRACING_V2', 'false')}")
    logger.info(f"📊 Health: http://localhost:{port}/health")
    logger.info(f"🧪 LangSmith test: http://localhost:{port}/langsmith-test")
    
    app.run(host="0.0.0.0", port=port, debug=False)