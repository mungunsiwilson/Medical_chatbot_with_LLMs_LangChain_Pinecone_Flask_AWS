import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# ============ LANGSMITH SETUP ============
# Set these BEFORE any other imports
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "medical-chatbot"

print(f"🔍 LangSmith Tracing: {'✅ Enabled' if os.environ.get('LANGCHAIN_TRACING_V2') == 'true' else '❌ Disabled'}")
print(f"🔑 LangSmith API Key: {'✅ Set' if os.environ.get('LANGCHAIN_API_KEY') else '❌ Not Set'}")
# =========================================

from flask import Flask, render_template, request, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from src.prompt import *
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Initialize components
embeddings = download_embeddings()
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_model = ChatGroq(model="llama-3.3-70b-versatile")

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
        print(f"🆕 New session: {session['session_id'][:8]}")
    
    get_or_create_memory(session['session_id'])
    return render_template('chat.html')

@app.route('/get', methods=["POST"])
def chat():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return "Please refresh the page."
        
        msg = request.form['msg']
        print(f"\n📥 User ({session_id[:8]}): {msg}")
        
        memory = get_or_create_memory(session_id)
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # ============ VERSION-AGNOSTIC TRACING ============
        # Method that works with all LangSmith versions
        import langsmith
        from langsmith import trace
        
        # Prepare metadata
        metadata = {
            "session": session_id[:8],
            "memory_messages": len(memory.chat_memory.messages),
            "query": msg[:100]
        }
        
        # Check LangSmith version and use appropriate method
        try:
            # Try new version API first
            with trace(
                name="Medical_Chat_Query",
                run_type="chain",
                metadata=metadata,
                tags=["medical", f"session_{session_id[:4]}"]
            ):
                # Execute RAG chain
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
                
        except TypeError as e:
            # Fallback: Old version without metadata parameter
            print(f"⚠️ Using fallback tracing (old API): {e}")
            
            # Just trace without metadata
            with trace(name="Medical_Chat", run_type="chain"):
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
        # ==================================================
        
        # Save to memory
        memory.save_context({"input": msg}, {"output": answer})
        
        print(f"🤖 Bot: {answer[:100]}...")
        print(f"🧠 Memory now has {len(memory.chat_memory.messages)} messages")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return "I'm having trouble processing your request."

@app.route('/debug_trace')
def debug_trace():
    """Test LangSmith tracing"""
    import langsmith
    from langsmith import trace
    
    print(f"LangSmith version: {langsmith.__version__}")
    
    # Test different methods
    test_results = []
    
    # Test 1: Simple trace
    try:
        with trace(name="Test_Trace_1", run_type="chain"):
            test_results.append("✅ Simple trace works")
    except Exception as e:
        test_results.append(f"❌ Simple trace failed: {e}")
    
    # Test 2: Trace with metadata
    try:
        with trace(name="Test_Trace_2", run_type="chain", metadata={"test": "true"}):
            test_results.append("✅ Trace with metadata works")
    except Exception as e:
        test_results.append(f"❌ Trace with metadata failed: {e}")
    
    return {
        "langsmith_version": langsmith.__version__,
        "api_key_set": bool(os.environ.get("LANGCHAIN_API_KEY")),
        "tracing_enabled": os.environ.get("LANGCHAIN_TRACING_V2") == "true",
        "test_results": test_results
    }

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=5000, debug=True)