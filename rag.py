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
import confluence_api 
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from convert_documents import split_text, load_documents
from typing import List
from langchain_core.runnables import RunnableParallel, RunnableLambda
from operator import itemgetter
        
# load .env Variablen 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chroma DB
#CHROMA_PATH = "chroma"

# hyperparameters
MAX_NUMBER_OF_RESULTS = 10
THRESHHOLD_SIMILARITY = 0.9




relevant_sources = ""

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

# template to control for hallucination and contradictions
rag_template = """
    Beantworte die Frage basierend auf dem gegebenen Kontext: {context}
    
    Wenn es Widersprüche im Kontext gibt, gebe unbedingt alle Wiedersprüche aus die, die Frage beantworten.      
    Wenn das Beantworten der Frage nicht möglich ist durch den gegebenen Kontext oder kein Kontext gegeben wurde, antworte IMMER "Ich weiß es nicht". 
    """

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]) 

# rewrite query given the chat_history
rewriting_template = """
    Ergänze "{input}" nur mit Informationen aus der Konversation, und auch nur wenn diese benötigt werden für die Beantwortung. Gib nur den transformierten Input aus.
    
    """

rewriting_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name = "chat_history"), 
                                                   ("user", rewriting_template)
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

# only rewrite with a given chat history
def route(info):
    if len(info["chat_history"])!=0:
        return rewriting
    else:
        return no_rewriting

# whole chain that gets invoked 
rag_chain = route | retrieval | rag_prompt | model | parser


def generate_response(message, history_list, stream=True):
    chat_history =[]
    for mes in history_list:
        # verschiedene Wrapper für verschiedene roles
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        chat_history.append(m)
    
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
    key = f"user_score"
    client.create_feedback(run_id, key=key, score=score)


def update_data():
    # update txt file
    print("loading new data")
    confluence_api.get_data_confluence()
    print("new data loaded from confluence")
    # upsert new data to pinecone
    global vectorstore
    # delete all existing vectors in pinecone vectorstore
    delete_vecs_pinecone()
    print("delete existing vectors")
    # load existing documents from json
    documents = load_documents(file="Data/confluence_data.json")
    # split text
    knowledge_chunks =  split_text(documents)
    # set up vector database and overwrite varible vectorstore with new one
    print("upserting data to pinecone")
    vectorstore = PineconeVectorStore.from_documents(
        knowledge_chunks, embeddings, index_name=index_pinecone)
    print("vectorstore is updated")


def delete_vecs_pinecone():
    # Langchanin doesn't sopport deleting all vectors
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Delete the index
    pc.delete_index(index_pinecone)
    # create new index with the same name 
    pc.create_index(index_pinecone, 
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )


# for debugging purposes
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




# def get_chunks_from_chroma(message):
#     # Prepare the DB.
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

#     # Search the DB for relevant chunks for answering the question
#     results = db.similarity_search_with_relevance_scores(message, k=MAX_NUMBER_OF_RESULTS)
    
#     # Filter relevant chunks to only the chunks with a simalarity greater than treshhold 
#     filtered_results = [(doc, score) for doc, score in results if score >= THRESHHOLD_SIMILARITY]

#     return filtered_results

         