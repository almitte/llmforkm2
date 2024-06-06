from confluence_api import get_data_confluence
from data_handling import load_documents, split_text
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os 
from dotenv import load_dotenv

load_dotenv()

def delete_vecs_pinecone():
    # langchanin doesn't sopport deleting all vectors
    # initialize pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # delete the index
    index_pinecone = "llm-km"
    pc.delete_index(index_pinecone)
    # create new index with the same name 
    pc.create_index(index_pinecone, 
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
    
def upsert_data_to_pinecone():
    # load existing documents from json
    path = "Data/confluence_data.json"
    documents = load_documents(file=path)
    # split text from convert_documents
    knowledge_chunks =  split_text(documents)
    
    # load new chunks to pinecone vectorstore,
    PineconeVectorStore.from_documents(knowledge_chunks, OpenAIEmbeddings(), index_name="llm-km")
        
