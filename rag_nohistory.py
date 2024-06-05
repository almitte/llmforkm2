import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate 
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from langchain.callbacks import LangChainTracer
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from operator import itemgetter
from typing import Literal, Generator 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
        
# load .env Variablen 
load_dotenv()

# tracing with Langsmith from Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-project"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
client = Client()

def initialize_chain():
    # template to control for hallucination and contradictions
    rag_template = """
        FRAGE: {input} 
        
        KONTEXT: {context}
        
        ANWEISUNG: 
        Beantworte die FRAGE basierend auf dem gegebenen KONTEXT.  
        Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
        
        """

    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    # inistialize specific model with api key and temperature    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    model1 = ChatGroq(temperature=0,groq_api_key=GROQ_API_KEY, model_name="llama3")
    # parses AIMessage to string
    parser = StrOutputParser() 

    # for embedding of chunks 
    embeddings = OpenAIEmbeddings()

    # initialize vectorstore
    index_pinecone = "llm-km"
    vectorstore = PineconeVectorStore(index_name=index_pinecone, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    multi_prompt = ChatPromptTemplate.from_template(
        template="""Sie sind ein KI-Sprachmodellassistent. Ihre Aufgabe besteht darin, fünf verschiedene Versionen der gegebenen Benutzerfrage zu generieren, 
        um relevante Dokumente aus einer Vektordatenbank abzurufen. Durch die Generierung mehrerer Perspektiven auf die Benutzerfrage besteht Ihr Ziel darin, 
        dem Benutzer dabei zu helfen, einige der Einschränkungen der distanzbasierten Ähnlichkeitssuche zu überwinden. Stellen Sie diese alternativen Fragen 
        durch Zeilenumbrüche getrennt bereit. 
        Ursprüngliche Frage: {input}""")
    
    llm = ChatOpenAI(temperature=0)

    # Chain
    generate_queries = multi_prompt | model | parser | (lambda x: x.split("\n"))
    
    # 2 functions for chaining:
    # compose page content of Documents to one string
    def format_docs(docs):
        global relevant_docs
        relevant_docs = docs
        return "\n\n".join([d.page_content for d in docs])

    from langchain.load import dumps, loads

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]
    
    retrieval_chain = generate_queries | retriever.map() | get_unique_union | format_docs
    
    # rag chain that gets invoked when you generate a response
    rag_chain = {"context": retrieval_chain, "input": itemgetter("input")}  | rag_prompt | model | parser
    
    return rag_chain


def generate_response(message: str, history_list: list[dict], stream: bool = True) -> str | Generator[str, None, None]:
    chat_history =[]
    # transform the chat history to langchain chat_history list
    # wrapping messages with Human and AIMessage
    for mes in history_list:
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        chat_history.append(m)   
    chain = initialize_chain()     
    # option to choose streaming or getting the entire response at once
    if stream:
        response = chain.stream({
            "chat_history": chat_history, 
            "input": message
            })
    else: 
        response = chain.invoke({
            "chat_history": chat_history, 
            "input": message
            })       
    return response
     

def get_relevant_sources() -> list[tuple[str, str]]:
    confluence_spacekey=os.getenv("LANGCHAIN_API_KEY")
    relevant_sources = []
    # relevant_docs are the chunks, each wrapped in a Document object
    for doc in relevant_docs:
        # put together the link to page of the chunk
        source = f"https://llmgruppenarbeit.atlassian.net/wiki/spaces/{confluence_spacekey}/pages/" + doc.metadata.get("p_id")
        title = doc.metadata.get("p_title")
        # tuple of the title and link of the chunk
        new_dic = (title, source)
        # no duplicates allowed
        if new_dic not in relevant_sources:
            relevant_sources.append(new_dic) 
    return relevant_sources
        

# send feedback (1 or 0) to tracer langsmith 
def send_feedback(run_id: str, score: Literal[0,1]):
    key = f"user_score_{run_id}"
    client.create_feedback(run_id, key=key, score=score)

# for debugging purposes
if __name__ == "__main__":
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input, chat_history, stream=True)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        for chunk in response:
            print(chunk, end="", flush=True)
        print("")