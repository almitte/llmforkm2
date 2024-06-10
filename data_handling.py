
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import json

    
# metadata extraction function for JSONLoader 
def extract_metadata(record: dict, metadata: dict) -> dict:

    metadata["p_title"] = record.get("p_title")
    metadata["p_id"] = record.get("p_id")
    metadata["p_parent"] = record.get("p_parent")
    metadata["last_edited"] = record.get("last_edited")
    metadata["sentence_window"] = record.get("sentence_window")

    return metadata  
  

def load_documents(file):
    """
    Loading documents from json file from data path and returning documents    
    
    """
    loader=JSONLoader(file,
                      jq_schema='.pages[]',
                      content_key="text",
                      metadata_func=extract_metadata)
    documents = loader.load()
    return documents

def split_text(documents: list[Document], chunk_size:int, chunk_overlap:int) -> list[Document]:
    """
    Function splits each individual document in smaller chunks so the context window for the prompt is not too long
    
    Args:
        documents (list): A list of Document objects to be split

    Returns:
        list: A list of chunks obtained from splitting the text of the documents
    
    """
  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def sentence_window_json(docs):
    data = [{doc.metadata["sentence_window"]: doc.page_content} for doc in docs]
    with open("Data/index_windows.json", "w") as f:
        json.dump(data, f)
    
    
    
    

    
