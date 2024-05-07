import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate 
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langsmith import Client
from langchain.callbacks import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import data 
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import re
from convert_documents import split_text, load_documents
        
# load .env Variablen 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chroma DB
#CHROMA_PATH = "chroma"

# hyperparameters
MAX_NUMBER_OF_RESULTS = 10
THRESHHOLD_SIMILARITY = 0.8
chunk_size = 1000
chunk_overlap= 20



relevant_sources=""

# tracing mit Langsmith from Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-project"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
client = Client()

# konvertiert AI Message zu einem String
parser = StrOutputParser() 

# Embedding der Wissenschunks
embeddings = OpenAIEmbeddings()

# Initialize vectorstore
index_pinecone = "llm-km"
vectorstore = PineconeVectorStore(index_name=index_pinecone, embedding=embeddings)

# template prompt
template = """
    Beantworte die Frage basierend auf dem gegebenen Kontext: {context}
            
    Wenn das Beantworten der Frage nicht möglich ist durch den gegebenen Kontekt, antworte IMMER "Ich weiß es nicht". 
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
    ])        

def get_chunks_from_pinecone(message):
    results = vectorstore.similarity_search_with_relevance_scores(message, k=MAX_NUMBER_OF_RESULTS)
    filtered_results = [(doc, score) for doc, score in results if score >= THRESHHOLD_SIMILARITY]
    return filtered_results


def initialize_chain():
    
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

    chain = prompt | model | parser 
    return chain

def generate_response(message, history_list, chain):

    filtered_results = get_chunks_from_pinecone(message)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    history =[]
    for mes in history_list:
        # verschiedene Wrapper für verschiedene roles
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        history.append(m)
    
    global relevant_sources 
    relevant_sources = []
    for doc, _score in filtered_results:
        source = "https://llmgruppenarbeit.atlassian.net/wiki/spaces/KB/pages/" + doc.metadata.get("p_id")
        title = doc.metadata.get("p_title")
        new_dic = (title, source)
        if new_dic not in relevant_sources:
            relevant_sources.append(new_dic) 
        

    # relevant_sources = list(set(relevant_sources))
   
    # .stream statt .invoke um Antwort zu streamen
    return chain.stream({
        "history": history, 
        "question": message,
        "context": context_text
        })


def send_feedback(run_id, score):
    key =f"feedback_{run_id}"
    client.create_feedback(
        run_id,
        key=key,
        score=score,
        )


def get_chunks_from_chroma(message):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB for relevant chunks for answering the question
    results = db.similarity_search_with_relevance_scores(message, k=MAX_NUMBER_OF_RESULTS)
    
    # Filter relevant chunks to only the chunks with a simalarity greater than treshhold 
    filtered_results = [(doc, score) for doc, score in results if score >= THRESHHOLD_SIMILARITY]

    return filtered_results

def update_data():
    # update txt file
    print("loading new data")
    data.get_data_confluence()
    print("new data loaded from confluence")
    print("upsert data to pinecone")
    # upsert new data to pinecone
    get_pinecone_with_new_data()
    print("vectorstore is updated")

def get_pinecone_with_new_data():
    global vectorstore
    # delete all existing vectors in pinecone vectorstore
    delete_vecs_pinecone()
    documents = load_documents(file="Data/confluence_data.json")
    knowledge_chunks =  split_text(documents)
    # set up vector database
    vectorstore = PineconeVectorStore.from_documents(
        knowledge_chunks, embeddings, index_name=index_pinecone)

def delete_vecs_pinecone():
    # Initialize Pinecone
    # Langchanin doesn't sopport deleting all vectors
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
   
    # Delete the index
    pc.delete_index(index_pinecone)

    # create new index
    pc.create_index(index_pinecone, 
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )

# Einfluss des Chatverlaufs verringer bei der Abfrage der Vektor-Datenbank (REWRITING)
# retriever = vectorStore.as_retriever(search_kwargs={"k":3})
# retriever_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name = "history"), 
#                                                     ("human", {input}),
#                                                     ("human", "Mit der gegebenen Konversation, generiere eine Suchabfrage zum Nachschlagen, um Informationen zu erhalten, die für die Konversation relevant sind.")])

# history_aware_retriever = create_history_aware_retriever(
#     llm=model, retriever=retriever, prompt=retriever_prompt
# )

