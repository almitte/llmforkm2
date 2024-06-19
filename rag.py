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
import json
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel, Field
from langchain_core.output_parsers import BaseOutputParser
from typing import List
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
        
# load .env Variablen 
load_dotenv()

# tracing with Langsmith from Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
client = Client()

#---------------------------------------------------------------------------------------------

# template to control for hallucination and contradictions
rag_template = """
    FRAGE: {input} 
    
    KONTEXT: {context}
    
    ANWEISUNG: 
    Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
    Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
    Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
    
    """
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
    ]) 

# rewrite query given the chat_history
rewriting_template = """
    Angesichts des gegebenen Chatverlaufs und der neuesten Benutzerfrage 
    die auf den Kontext im Chatverlauf verweisen könnte, formulieren Sie eine eigenständige Frage 
    was auch ohne den Chatverlauf nachvollziehbar ist. Beantworten Sie die Frage NICHT, 
    Formulieren Sie es bei Bedarf einfach um und geben Sie es ansonsten so zurück, wie es ist.
    """
rewriting_prompt = ChatPromptTemplate.from_messages([
    ("system", rewriting_template),
    MessagesPlaceholder(variable_name = "chat_history"), 
    ("human", "{input}")
    ])

#---------------------------------------------------------------------------------------------

# inistialize specific model with api key and temperature    
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

# parses AIMessage to string
parser = StrOutputParser() 

# for embedding of chunks 
embeddings = OpenAIEmbeddings()

# initialize vectorstores, large for basic and chat history chain and small one for sentence window
index_pinecone_sm = "llm-km-sm"
vectorstore_sm = PineconeVectorStore(index_name=index_pinecone_sm, embedding=embeddings)
index_pinecone_lg = "llm-km"
vectorstore_lg = PineconeVectorStore(index_name=index_pinecone_lg, embedding=embeddings)
    
# hyperparameters for retriever 
MAX_NUMBER_OF_RESULTS = 10
THRESHHOLD_SIMILARITY = 0.9

# specifiy vectorstore and parameters for retriever 
retriever_sm = vectorstore_sm.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": MAX_NUMBER_OF_RESULTS, "score_threshold": THRESHHOLD_SIMILARITY})
retriever_lg = vectorstore_lg.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": MAX_NUMBER_OF_RESULTS, "score_threshold": THRESHHOLD_SIMILARITY})

def format_docs(docs):
    """
    Führt den Seiteninhalt einer Liste von Dokumentobjekten zusammen.

    Parameter:
    docs (list): Eine Liste von Dokumentobjekten. Jedes Dokumentobjekt muss 
                 ein `page_content` Attribut haben, das den Inhalt des 
                 Dokuments als String enthält.

    Rückgabewert:
    str: Ein einzelner String, der den zusammengefügten Seiteninhalt aller 
         Dokumente enthält, wobei der Inhalt jedes Dokuments durch zwei 
         Zeilenumbrüche getrennt ist.
    """
    global relevant_docs
    relevant_docs = docs
    return "\n\n".join([d.page_content for d in docs])


def build_basic_chain():
    """
    Reasoning: Basic oder naives RAG führt direkt mithilfe der Nutzeranfrage ein Retrieval durch und nutzt die retrievten Dokumente als Kontext zur Beantwortung der Frage.
    
    Chain-Komponenten:
    - retrieval_lg: Nutzt die Nutzeranfrage für Retrieval
    - rag_prompt_no_history: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort
    
    Quellen:
    - 

    """

    ### Prerequisites
    rag_template = """
    FRAGE: {input} 
    
    KONTEXT: {context}
    
    ANWEISUNG: 
    Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
    Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
    Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
    """
    rag_prompt_no_history = ChatPromptTemplate.from_messages([("system", rag_template),("human", "{input}")]) 

    ### Erstellen der Chain    
    chain = {"context": retriever_lg | format_docs, "input": RunnablePassthrough()} | rag_prompt_no_history | model | parser 

    return chain


def get_rewriting_chain():
    """
    Reasoning: Rewriting löst ein grundlegenes Problem von naivem RAG: oftmals stellen Nutzer kontextsensitive Fragen, die sich auf vorhergehende Fragen beziehen. 
    Ohne die vorhergehende Frage zu kennen, kann mithilfe dieser generischen weiterführenden Frage kein optimales Retrieval durchgeführt werden. Deswegen wird beim Rewriting ein LLM
    beaufragt, die aktuelle Nutzeranfrage dahingehend zu evaluieren, ob sie sich auf eine vorhegehende Frage bezieht und falls dies zutrifft, die Nutzeranfrage umzuschreiben.

    Bsp.: 
    - Nutzeranfrage 1: Wer ist Obiwan?
    - Nutzeranfrage 2: Wer ist SEIN Padwan?
        - LLM evaluiert: Nutzeranfrage 2 bezieht sich auf Nutzeranfrage 1 > Rewriting
            - Neue Nutzeranfrage 2: Wer ist Obiwans Padawan?
    
    Alternativen: Alternativ zum LLM Ansatz können hierfür Coreference Resolution (CR) Ansätze des NLPs genutzt werden. Diese identifizieren alle linguistischen Terme, die sich auf
    den selben realweltlichen Gegenstand beziehen und mithilfe derer ein Austausch bspw. von possesiv Pronomen (sein, ihr) durchgeführt werden kann. 
    Quelle: https://towardsdatascience.com/intro-to-coreference-resolution-in-nlp-19788a75adee    
    
    Chain-Komponenten:
    - route_rewriting: Funktion welche evaluiert, ob eine Chat History besteht und darauf basierend rewriting oder no_rewriting zurückgibt
        - rewriting: RunnableParallel Komponente, welche eine Anfrage an ein LLM geschickt, um die Nutzerfrage umzuschreiben, falls ein Bezug zu vorhergehenden Fragen besteht
        - no_rewriting: RunnablePassthrough Komponente, welche die Parameter an die nächste Komponente weiterleitet 
    - retrieval: RunnableParallel Komponente, um das Retrieval für die Nutzerfrage zu realisieren
    - rag_prompt: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort

    Quellen:
    - 

    """
    
    ### Prerequisites
    rag_template = """
        FRAGE: {input} 
        
        KONTEXT: {context}
        
        ANWEISUNG: 
        Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
        Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
        Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
        """
    rag_prompt = ChatPromptTemplate.from_messages([("system", rag_template),MessagesPlaceholder(variable_name="chat_history"),("human", "{input}")]) 

    rewriting_template = """
        Angesichts des gegebenen Chatverlaufs und der neuesten Benutzerfrage 
        die auf den Kontext im Chatverlauf verweisen könnte, formulieren Sie eine eigenständige Frage 
        was auch ohne den Chatverlauf nachvollziehbar ist. Beantworten Sie die Frage NICHT, 
        Formulieren Sie es bei Bedarf einfach um und geben Sie es ansonsten so zurück, wie es ist.
        """
    rewriting_prompt = ChatPromptTemplate.from_messages([("system", rewriting_template),MessagesPlaceholder(variable_name = "chat_history"),("human", "{input}")])

    def route_rewriting(dic):
        if len(dic["chat_history"])!=0: return rewriting
        else: return no_rewriting

    ### Erstellen der Chain        
    rewriting = RunnableParallel({"input": rewriting_prompt | model | parser, "chat_history": itemgetter("chat_history")})
    no_rewriting = RunnablePassthrough()
    retrieval = RunnableParallel({"context": itemgetter("input") | retriever_lg | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")})
    chain = route_rewriting | retrieval | rag_prompt | model | parser
    
    return chain 


def get_sentence_window_chain():
    # 2 functions for chaining:
    # compose page content of Documents to one string
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs]) # "Titel: " + d.metadata["p_title"] + " (Zuletzt Geändert: " +  d.metadata["last_edited"] +")" + "\n" + 

    def route_rewriting(dic):
        if len(dic["chat_history"])!=0: return rewriting
        else: return no_rewriting
    
    # returns the bigger an index to the corresponding bigger chunk of smaller chunk
    def get_window_index(docs):
        global relevant_docs
        relevant_docs = docs
        return set([doc.metadata["sentence_window"] for doc in docs])
    
    def get_docs(indexes):
        with open('Data/index_windows.json') as f:
            index2chunk = json.load(f)
            
            def search(i):
                value = None
                for item in index2chunk:
                    if i in item:
                        return Document(item[i])
                    
            return [search(index) for index in indexes]
    
    # RunnableParallel so all value "chains" of the dictionary get parallel excecuted, itemgetter to get the values of keys from the input dictionary
    retrieval = RunnableParallel({"context": itemgetter("input") | retriever_sm | RunnableLambda(get_window_index) |  RunnableLambda(get_docs) | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")})

    # rewrite the query given the chat history
    rewriting = RunnableParallel({"input": rewriting_prompt | model | parser, "chat_history": itemgetter("chat_history")})

    # do nothing and pass on the entire dictionary 
    no_rewriting = RunnablePassthrough()

    # rag chain that gets invoked when you generate a response
    chain = route_rewriting | retrieval | rag_prompt | model | parser
    return chain

def get_individual_subquestions_chain():
    """
    Reasoning: Wenn ein Nutzer eine Frage stellt, gibt es keine Garantie, dass die relevanten Ergebnisse mit einer einzigen Abfrage zurückgegeben werden können. 
    Vor allem wenn die Frage aus mehreren unabhängigen Teilfragen besteht, ist ein einziges Retrieval wenig zielführend.
    Deshalb ist es sinnvoll, für manche Anwendungsfälle die Ausgangsfrage zunächst in kleinere Teilfragen zu zerlegen, mit diesen ein einzelnes Retrieval durchzuführen und 
    dann die Ausgangsfrage mithelfe des Kontextes zu beantworten.

    Chain-Komponenten:
    - retrieval: RunnableParallel Komponente, um das Retrieval für die Nutzerfrage zu realisieren
        - retriever_lg_multi: MultiQueryRetriever, welcher für alle Teilfragen ein Retrieval durchführt
            - llm_chain: LLMChain, welche Teilfragen aus der Ausgangsfrage generiert
    - rag_prompt: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort

    Quellen:
    - https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/

    """

    ### Prerequisites
    subquestion_template = """
    Sie sind ein KI-Sprachmodell-Assistent. Ihre Aufgabe ist es, die Ausgangsfrage zu analysieren und die einzelnen Teilfragen/Teilprobleme zurückzugeben,\n 
    um relevante Dokumente aus einer Vektordatenbank zu finden. Geben Sie diese Teilfragen Fragen durch Zeilenumbrüche getrennt ein.\n
    Ausgangsfrage: {question} 
    """
    subquestion_prompt = PromptTemplate.from_template(subquestion_template)

    rag_template = """
    FRAGE: {original_question} 
   
    KONTEXT: {context}
    
    ANWEISUNG: 
    Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
    Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
    Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
    """
    rag_prompt = ChatPromptTemplate.from_messages([("system", rag_template),MessagesPlaceholder(variable_name="chat_history"),("human", "{original_question}")]) 

    class LineListOutputParser(BaseOutputParser[List[str]]):
        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return lines
    output_parser = LineListOutputParser()

    ### Erstellen der Chain
    llm_chain = LLMChain(llm=model, prompt=subquestion_prompt, output_parser=output_parser)
    retriever_lg_multi = MultiQueryRetriever(retriever=retriever_lg, llm_chain=llm_chain)
    retrieval = RunnableParallel({"context": itemgetter("input") | retriever_lg_multi | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history"), "original_question": itemgetter("original_question")})
    chain = retrieval | rag_prompt | model | parser
    
    return chain 


def get_consecutive_subquestions_chain(message):
    """
    Reasoning: Die Annahme bei Consecutive Subquestions ist, dass für manche Anfragen es notwendig ist, dass die Ausgangsfrage zunächst in aufeinanderaufbauende Teilfragen zerlegt
    und diese dann in solcher Art nacheinander abgearbeitet werden müssen, dass die Antwort der vorhergehenden Teilfrage zur Beantwortung der nächsten Teilfrage notwendig ist.
    Dieser Ansatz verfolgt dabei einen iterativen Aufbau, dass zunächst die Teilfragen generiert werden und nachfolgend die Teilfragen mithilfe der retrievten Dokumente der aktuellen 
    Teilfrage und den Antworten der vorhergehenden Teilfragen beantwortet werden. Schlussendlich wird über die Beantwortung der Teilfragen eine Beantwortung der Ausgangsfrage generiert.

    Chain-Komponenten:
    - retrieval: RunnableParallel Komponente, um das Retrieval für die Nutzerfrage zu realisieren
        - retriever_lg_multi: MultiQueryRetriever, welcher für alle Variationen der Nutzerfrage ein Retrieval durchführt
            - llm_chain: LLMChain, welche Variationen der Nutzerfrage aus der Ausgangsfrage generiert
    - rag_prompt: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort

    Quellen:
    - https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/hyde/
    - 

    """

    ### Prerequisites
    consecutive_subquestions_template = """
    Sie sind ein hilfreicher Assistent, der mehrere Teilfragen zu einer Ausgangsfrage erstellt. \n
    Das Ziel ist es, die Eingabe in eine Reihe von Teilproblemen/Teilfragen zu zerlegen, die isoliert beantwortet werden können. \n
    Generiere bitte mehrere Suchanfragen in Bezug auf diese Ausgangsfrage: {input} \n
    Erstellen Sie Teilfragen, deren Antworten zur Beantwortung der obigen Frage erforderlich sind. Geben Sie NUR diese Fragen in folgendem Format zurück: 'QX - Frage (Zeilenumbruch)'.
    """
    consecutive_subquestions_prompt = ChatPromptTemplate.from_messages([("system", consecutive_subquestions_template),("human", "{input}")])

    template = """Hier ist die Frage die du beantworten musst:

    \n --- \n {question} \n --- \n

    Hier sind alle verfügbaren Backgroundinformationen als Q&A Paare:

    \n --- \n {q_a_pairs} \n --- \n

    Hier zusätzlicher Kontext der für die Beantwortung der Frage relevant ist: 

    \n --- \n {context} \n --- \n

    Beantworte die Frage anhand des obigen Kontexts und der Frage-Antwort-Paare, die im Hintergrund stehen.: \n {input}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    generate_queries_decomposition = (consecutive_subquestions_prompt | model | StrOutputParser() | (lambda x: x.split("\n")))
    questions = generate_queries_decomposition.invoke({"input": message})

    def format_qa_pair(question, answer):       
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    q_a_pairs = ""
    for idx, q in enumerate(questions):
        
        rag_chain = (
        {"context": itemgetter("question") | retriever_lg | format_docs, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs"),
        "input": itemgetter("input")} 
        | decomposition_prompt
        | model
        | StrOutputParser())

        if idx == len(questions) - 1:
            return rag_chain.stream({"question":q,"q_a_pairs":q_a_pairs,"input":message})

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs,"input":message})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

def get_hyde_chain():
    """
    Reasoning: Die Annahme von HyDE (Hypothetical Documents Embedding) ist, dass sich Nutzerfragen/-anfragen und gespeicherte Dokumente in ihrer Art grundsätzlich unterscheiden
    (Dokumente sind länger, beinhalten Noise und sind anders formuliert), wodurch Nutzerfragen für ein gezieltes Retrieval mithilfe von Embeddings ungeeignet sind. 
    Deshalb werden wird bei HyDE ein LLM beauftragt, zur Beantwortung der Nutzerfrage zunächst ein hypothetisches Dokument zu formulieren, welches dann statt der Nutzerfrage 
    für das Retrieval relevanter Dokumente verwendet wird. Dadurch erhofft man sich eine verbessertes Retrieval.

    Chain-Komponenten:
    - route_rewriting: Funktion welche evaluiert, ob eine Chat History besteht und darauf basierend rewriting oder no_rewriting zurückgibt
        - rewriting: RunnableParallel Komponente, welche eine Anfrage an ein LLM geschickt, um die Nutzerfrage umzuschreiben, falls ein Bezug zu vorhergehenden Fragen besteht
        - no_rewriting: RunnablePassthrough Komponente, welche die Parameter an die nächste Komponente weiterleitet 
    - hyde: RunnableParallel Komponente, welcher aus der Nutzerfrage ein hypothetisches Dokument generiert und dieses als Input an das Retrieval weiterleitet
    - retrieval: RunnableParallel Komponente, 
    - rag_prompt: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort

    Quellen:
    - https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/hyde/
    - 

    """

    ### Prerequisites
    hyde_template = """
    Bitte schreib ein Confluence Wissensbasis Eintrag, um die Nutzer Frage zu beantworten 
    Frage: {input}"""
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

    rag_template = """
    FRAGE: {original_question} 
    
    KONTEXT: {context}
    
    ANWEISUNG: 
    Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
    Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
    Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
    """ 
    rag_prompt_2 = ChatPromptTemplate.from_messages([("system", rag_template),MessagesPlaceholder(variable_name="chat_history"),("human", "{original_question}")]) 

    def route_rewriting(dic):
        if len(dic["chat_history"])!=0: return rewriting
        else: return no_rewriting

    ### Erstellen der Chain
    rewriting = RunnableParallel({"input": rewriting_prompt | model | parser, "chat_history": itemgetter("chat_history"), "original_question": itemgetter("original_question")})
    no_rewriting = RunnablePassthrough()
    hyde = RunnableParallel({"input": hyde_prompt | model | parser, "chat_history": itemgetter("chat_history"), "original_question": itemgetter("original_question")})
    retrieval = RunnableParallel({"context": itemgetter("input") | retriever_lg | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history"), "original_question": itemgetter("original_question")})
    chain = route_rewriting | hyde | retrieval | rag_prompt_2 | model | parser

    return chain 


def get_multiquery_chain():
    """
    Reasoning: Die Annahme bei Multiquery (oder auch Multishot Ansatz) ist, dass sich Vektoren 2er Anfragen mit der gleichen Bedeutung aber unterschiedlicher Formulierung
    voneinander schwach bis mittelstark unterscheiden können. Dadurch können sich abhängig davon, wie der Nutzer eine Anfrage formuliert, die retrievten Dokumente unterscheiden.
    Deshalb wird beim Multiquery Ansatz ein LLM beauftragt, Variationen der Nutzerfrage zu generieren, mithilfe derer ein unabhängiges Retrieval durchgeführt und die Menge der 
    retrievten Dokumente anschließend zusammengeführt werden kann. Vom Aufbau ähnelt sich dieser Ansatz mit dem Subquestion Ansatz.

    Chain-Komponenten:
    - retrieval: RunnableParallel Komponente, um das Retrieval für die Nutzerfrage zu realisieren
        - retriever_lg_multi: MultiQueryRetriever, welcher für alle Variationen der Nutzerfrage ein Retrieval durchführt
            - llm_chain: LLMChain, welche Variationen der Nutzerfrage aus der Ausgangsfrage generiert
    - rag_prompt: Prompt Vorlage zur Beantwortung der Nutzerfrage mithilfe der retrievten relevanten Dokumente
    - model: OpenAI LLM welches zur Beantwortung der Nutzerfrage verwendet wird
    - parser: Verarbeiten der vom LLM erhaltenen Antwort

    Quellen:
    - https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/hyde/
    - 

    """

    ### Prerequisites
    multiquery_template = """
    Sie sind ein KI-Sprachmodell-Assistent. Ihre Aufgabe ist es, fünf verschiedene 
    verschiedene Versionen der gegebenen Benutzerfrage zu generieren, um relevante Dokumente aus einer Vektordatenbank 
    Datenbank zu finden. Indem Sie mehrere Perspektiven auf die Frage des Benutzers erzeugen, wollen Sie dem Benutzer helfen
    Ziel ist es, dem Benutzer zu helfen, einige der Einschränkungen der entfernungsbasierten Ähnlichkeitssuche zu überwinden. 
    Geben Sie diese alternativen Fragen durch Zeilenumbrüche getrennt ein. Ursprüngliche Frage: {question} 
    """
    multiquery_prompt = PromptTemplate.from_template(multiquery_template)

    rag_template = """
    FRAGE: {original_question} 
    
    KONTEXT: {context}
    
    ANWEISUNG: 
    Beantworte die FRAGE basierend auf dem gegebenen KONTEXT. 
    Wenn KONTEXT leer ist, anworte mit "Ich weiß es nicht". 
    Wenn das Beantworten der FRAGE nicht möglich ist durch den gegebenen KONTEXT, antworte immer "Ich weiß es nicht".
    """        
    rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{original_question}")
    ]) 

    class LineListOutputParser(BaseOutputParser[List[str]]):

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return lines
    output_parser = LineListOutputParser()

    ### Erstellen der Chain
    llm_chain = LLMChain(llm=model, prompt=multiquery_prompt, output_parser=output_parser)
    retriever_lg_multi = MultiQueryRetriever(retriever=retriever_lg, llm_chain=llm_chain)
    retrieval = RunnableParallel({"context": itemgetter("input") | retriever_lg_multi | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history"), "original_question": itemgetter("original_question")})
    chain = retrieval | rag_prompt | model | parser

    return chain

def chain_to_initiate(chain_type: str):
    """
    Erstellt die entsprechende Chain basierend auf der Nuterauswahl

    Parameter:
    chain_type (str): Die Art der ausgewählten Chain

    Rückgabewert:
    Die Chain basierend auf der entsprechenden Chain-Funktion.
    
    """

    match chain_type:
        case "basic": return build_basic_chain()
        case "history": return get_rewriting_chain()
        case "window": return get_sentence_window_chain()
        case "individual_subquestions": return get_individual_subquestions_chain()
        case "consecutive_subquestions": return get_consecutive_subquestions_chain()
        case "hyde": return get_hyde_chain()
        case "multi": return get_multiquery_chain()

def has_chain_history(chain_type):
    """
    Evaluiert, ob Chain eine History nutzt und ob sie die Ausgangsfrage als zusätzlichen Parameter benötigt.

    Parameter:
    chain_type (str): Die Art der ausgewählten Chain

    Rückgabewert:
    Einen String der die Art der verwendeten Parameter angibt.
    
    """

    match chain_type:
        case "basic": return "basic"
        case "history": return "history_no_original_question"
        case "window": return "history_no_original_question"
        case "individual_subquestions": return "history_with_original_question"
        case "consecutive_subquestions": return "history_with_original_question"
        case "hyde": return "history_with_original_question"
        case "multi": return "history_with_original_question"

def generate_response(message: str, history_list: list[dict], chain_type: Literal["basic", "history", "window", "hyde", "consecutive_subquestions", "individual_subquestions", "multi"]) -> Generator[str, None, None]:
    """
    Generiert eine Antwort zu einer Nutzerfrage unter Zuhilfenahme unterschiedlicher Chains basierend auf dem chain_type.

    Parameter:
    message (str): Die Nutzeranfrage, die beantwortet werden soll.
    history_list (list[dict]): Eine Liste der bisherigen Nachrichten im Chat. 
                               Jede Nachricht ist ein Dictionary mit den Schlüsseln "role" und "content".
                               "role" kann entweder "user" oder "AI" sein.
    chain_type (Literal): Der Typ der Chain, die verwendet werden soll. 

    Rückgabewert:
    Generator[str, None, None]: Ein Generator, der die Antwort(en) in Form von Strings liefert.
    
    """ 
    
    chat_history =[]
    for mes in history_list:
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        chat_history.append(m)
    
    chain = chain_to_initiate(chain_type)
    history = has_chain_history(chain_type) 
    match history:
        case "basic": response = chain.stream(message)
        case "history_no_original_question": response = chain.stream({"chat_history": chat_history, "input": message})
        case "history_with_original_question": response = chain.stream({"chat_history": chat_history, "input": message, "original_question": message})
    
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
        response = generate_response(user_input, chat_history, "hyde")
        chat_history.append({"role": "user", "content": user_input})
        response_string = ""
        for chunk in response:
            print(chunk, end="", flush=True)
            response_string += chunk
        print("")
        chat_history.append({"role": "assistant", "content": response_string})

