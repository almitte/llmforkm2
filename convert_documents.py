
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import os
import shutil
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

DATA_PATH = "Data"
CHROMA_PATH = "chroma"

def main():
    ...
    #generate_data_store()

def load_documents():
    """
    Loading documents from txt files from data path and returning documents    
    
    """

    loader=DirectoryLoader(DATA_PATH, glob='**/*.txt')
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    """
    Function splits each individual document in smaller chunks so the context window for the prompt is not too long
    
    Args:
        documents (list): A list of Document objects to be split

    Returns:
        list: A list of chunks obtained from splitting the text of the documents
    
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_data_store():
    """
    Main entry point to generate data store by loading documents from storage, splitting them into chunks, and storing the chunks in a Chroma database.
    
    """

    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def save_to_chroma(chunks: list[Document]):
    """
    Saves chunks in a chroma database

    Args:
        chunks (list): A list of Document objects to be stored in the database
    
    """
    load_dotenv()

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, 
        OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY")), 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()



