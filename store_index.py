from dotenv import load_dotenv
import os 
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings 
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files("C:\\Users\\Mungunsi\\Desktop\\Medical_chat_bot\\Medical_chatbot_with_LLMs_LangChain_Pinecone_Flask_AWS\\data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
embeddings = download_embeddings()

#Initialize pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medicalchatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
#Load index

#embed each pine chunk and upset the embeddings into pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
    )