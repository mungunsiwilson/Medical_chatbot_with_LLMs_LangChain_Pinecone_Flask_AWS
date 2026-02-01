from flask import Flask, render_template, request, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Random secret key for sessions

load_dotenv()

# Initialize components
embeddings = download_embeddings()
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_model = ChatGroq(model="llama-3.3-70b-versatile")

# ============ MEMORY STORAGE ============
# Dictionary to store memory for each user session
session_memories = {}
# ========================================

def get_or_create_memory(session_id):
    """Get existing memory or create new one for session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]

@app.route('/')
def index():
    # Create unique session ID for each user
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Initialize memory for this session
    get_or_create_memory(session['session_id'])
    
    return render_template('chat.html')

@app.route('/get', methods=["POST"])
def chat():
    try:
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return "Please refresh the page to start a new session."
        
        # Get user message
        msg = request.form['msg']
        print(f"\n📥 User ({session_id[:8]}): {msg}")
        
        # Get memory for this session
        memory = get_or_create_memory(session_id)
        
        # ============ MEMORY INTEGRATION ============
        # 1. Get conversation history from memory
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # 2. Create prompt WITH memory
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),  # Memory goes here
                ("human", "{input}"),
            ]
        )
        
        # 3. Create chains
        question_answering_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answering_chain)
        
        # 4. Prepare input with memory
        chain_input = {
            "input": msg,
            "chat_history": chat_history  # Pass memory to chain
        }
        
        # 5. Get response
        response = rag_chain.invoke(chain_input)
        answer = response['answer']
        
        # 6. Save conversation to memory
        memory.save_context({"input": msg}, {"output": answer})
        # =============================================
        
        print(f"🤖 Bot: {answer[:100]}...")
        print(f"🧠 Memory now has {len(memory.chat_memory.messages)} messages")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return "I'm having trouble processing your request."

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear memory for current session"""
    session_id = session.get('session_id')
    if session_id and session_id in session_memories:
        session_memories[session_id].clear()
        return "Memory cleared!"
    return "No active session."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)