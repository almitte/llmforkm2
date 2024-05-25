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
from convert_documents import split_text, load_documents
from typing import List
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
        
# load .env Variablen 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chroma DB
#CHROMA_PATH = "chroma"

# hyperparameters
MAX_NUMBER_OF_RESULTS = 10
THRESHHOLD_SIMILARITY = 0.9




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
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": MAX_NUMBER_OF_RESULTS, "score_threshold": THRESHHOLD_SIMILARITY})

# template prompt
rag_template = """
    Beantworte die Frage basierend auf dem gegebenen Kontext: {context}
          
    Wenn das Beantworten der Frage nicht möglich ist durch den gegebenen Kontext oder kein Kontext gegeben wurde, antworte IMMER "Ich weiß es nicht". 
    """

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]) 

rewriting_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name = "chat_history"), 
                                                   ("user", "{input}"),
                                                   ("user", "Mit der gegebenen Konversation, generiere eine Suchabfrage zum Nachschlagen, um Informationen zu erhalten, die für die Konversation relevant sind. Gib nur die wirkliche Suchabfrage aus")
                                                   ])       

 # compose page content of Documents to one string
def format_docs(docs):
    global relevant_docs
    relevant_docs = docs
    return "\n\n".join([d.page_content for d in docs])
    
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

# RunnableParallel so both values of the dictionary get parallel excecuted, itemgetter to get the values of keys from the input dictionary
retrieval = RunnableParallel({"context": itemgetter("input") | retriever | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")})

# rewrite the query given the chat history
rewriting = RunnableParallel({"input": rewriting_prompt | model | parser, "chat_history": itemgetter("chat_history")})

# do nothing 
no_rewriting = RunnableParallel({"chat_history": itemgetter("chat_history"), "input": itemgetter("input")})


def generate_response(message, history_list, stream=True):
    chat_history =[]
    for mes in history_list:
        # verschiedene Wrapper für verschiedene roles
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        chat_history.append(m)
    
    # rewriting happens only with an existing chat history
    check_rewriting = no_rewriting if len(chat_history) == 0 else rewriting 

    rag_chain = check_rewriting| retrieval | rag_prompt | model | parser
    
    if stream:
        return rag_chain.stream({
            "chat_history": chat_history, 
            "input": message
            })
    else: return rag_chain.invoke({
            "chat_history": chat_history, 
            "input": message
            })

def get_relevant_sources():
    relevant_sources = []
    for doc in relevant_docs:
        source = "https://llmgruppenarbeit.atlassian.net/wiki/spaces/KB/pages/" + doc.metadata.get("p_id")
        title = doc.metadata.get("p_title")
        new_dic = (title, source)
        if new_dic not in relevant_sources:
            relevant_sources.append(new_dic) 
    return relevant_sources
        

def send_feedback(run_id, score):
    key =f"user_score"
    client.create_feedback(
        run_id,
        key=key,
        score=score,
        )


# def get_chunks_from_chroma(message):
#     # Prepare the DB.
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

#     # Search the DB for relevant chunks for answering the question
#     results = db.similarity_search_with_relevance_scores(message, k=MAX_NUMBER_OF_RESULTS)
    
#     # Filter relevant chunks to only the chunks with a simalarity greater than treshhold 
#     filtered_results = [(doc, score) for doc, score in results if score >= THRESHHOLD_SIMILARITY]

#     return filtered_results



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



if __name__ == "__main__":
    
    # Initialize chat history
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input, chat_history, stream=False)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        print(response)
         