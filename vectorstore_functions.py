from confluence_api import get_data_confluence
from data_handling import load_documents, split_text, sentence_window_json
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
    index_pinecone_lg = "llm-km"
    index_pinecone_sm = "llm-km-sm"
    pc.delete_index(index_pinecone_lg)
    pc.delete_index(index_pinecone_sm)
    # create new index with the same name 
    pc.create_index(index_pinecone_lg, 
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
    pc.create_index(index_pinecone_sm, 
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
    
def upsert_data_to_pinecone():
    # load existing documents from json
    path = "Data/confluence_data.json"
    documents = load_documents(file=path)
    # split text from convert_documents
    knowledge_chunks_lg =  split_text(documents, chunk_size=1000, chunk_overlap=500)
    chunk_id = 0
    for chunk_lg in knowledge_chunks_lg:   
        chunk_lg.metadata["sentence_window"]=str(chunk_id)
        chunk_id += 1
    
    knowledge_chunks_sm = split_text(knowledge_chunks_lg,chunk_size=200, chunk_overlap=50)
    sentence_window_json(knowledge_chunks_lg)  
    # load new chunks to pinecone vectorstore,
    PineconeVectorStore.from_documents(knowledge_chunks_sm, OpenAIEmbeddings(), index_name="llm-km-sm")
        
