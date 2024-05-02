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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import data 
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone
        
# load .env Variablen 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
vectorstore = PineconeVectorStore(index_name="llm-km", embedding=embeddings)

# template prompt
template = """
    Beantworte die Frage basierend auf dem gegebenen Kontext: {context}
            
            
    Wenn das Beantworten der Frage nicht möglich ist durch den gegebenen Kontekt, antworte "Ich weiß es nicht". 
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
    ])        


def initialize_chain(name):
    
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=name)

    chain = prompt | model | parser 
    return chain
    

def generate_response(message, history_list, chain):
    history =[]
    for mes in history_list:
        # verschiedene Wrapper für verschiedene roles
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        history.append(m)
        
    # .stream statt .invoke um Antwort zu streamen
    return chain.stream({
        "history": history, 
        "question": message,
        # Document datei 
        "context": vectorstore.similarity_search(message)[:3]
        })


def send_feedback(run_id, score):
    key =f"feedback_{run_id}"
    client.create_feedback(
        run_id,
        key=key,
        score=score,
        )

def update_data():
    # update txt file
    data.get_data_confluence()
    # upsert new data to pinecone
    get_pinecone_with_new_data()

def get_pinecone_with_new_data():
    # load .txt file as a Document (object from Langchain)
    knowledge_doc = TextLoader("confluence_data.txt")
    # only string needed
    knowledge_str = knowledge_doc.load()   
    # splitting text in chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    knowledge_chunks = text_splitter.split_documents(knowledge_str)
    # set up vector database
    vectorstore = PineconeVectorStore.from_documents(
        knowledge_chunks, embeddings, index_name="llm-km")


# Einfluss des Chatverlaufs verringer bei der Abfrage der Vektor-Datenbank (REWRITING)
# retriever = vectorStore.as_retriever(search_kwargs={"k":3})
# retriever_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name = "history")], 
#                                                     ("human", {input}),
#                                                     ("human", "Mit der gegebenen Konversation, generiere eine Suchabfrage zum Nachschlagen, um Informationen zu erhalten, die für die Konversation relevant sind."))

# history_aware_retriever = create_history_aware_retriever(
#     llm=model, retriever=retriever, prompt=retriever_prompt
# )

