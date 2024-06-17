import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate 
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from langchain.callbacks import LangChainTracer
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from operator import itemgetter
import pandas as pd 
import rag
from langchain import callbacks
        
# load .env Variablen 
load_dotenv()

# tracing with Langsmith from Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
client = Client()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

eval_template = """ 
    FRAGE: {input}
    MUSTERANTWORT: {answer}
    RAG-ANTWORT: {rag_answer}
    
    ANWEISUNG:
    Du bist ein Evaluations-Assistent für ein Retrieval-Augmentation (RAG) System. Vergleiche für die gegebene FRAGE die MUSTERANTWORT mit der RAG-ANTWORT. 
    Wenn die RAG-ANTWORT den Inhalt der MUSTERANTWORT beinhaltet, antworte nur mit "1". 
    Wenn die RAG-ANTWORT den Inhalt der MUSTERANTWORT nicht beihaltet, antworte nur mit "0".   
    """

eval_prompt = ChatPromptTemplate.from_template(eval_template)

model = ChatOpenAI(model="gpt-4", temperature=0)

parser = StrOutputParser()

def add_answer(answer):
    global current_answer
    current_answer = answer
    return answer

# can be switched to basic or only history chain 
rag_chain = rag.get_sentence_window_chain()
# handle evaluation chain input dictionary 
answer_chain = {"input": itemgetter("input"), "answer": itemgetter("answer"), "rag_answer": {"input": itemgetter("input"), "chat_history": lambda x: []} | rag_chain | add_answer}
# entire chain with evaluation prompt
eval_chain = answer_chain | eval_prompt | model | parser

# use it on a pandas dataframe and it adds 3 columns: rag answer, evaluation (1 for right, 0 for wrong) and the run id to check what went wrong in langsmith
def evaluate(df):
    # initialize columns
    df["rag_answer"] = None
    df["y/n"] = None
    df["run_id"] = None
    for index, row in df.iterrows():
        # track run id
        with callbacks.collect_runs() as cb:
            df.loc[index, "y/n"] = eval_chain.invoke({"input": row[0], "answer": row[1]})
            run_id = cb.traced_runs[0].id
        df.loc[index, "rag_answer"] = current_answer
        df.loc[index, "run_id"] = run_id
    # calculate accuracy of the test questions and print it to console
    accuracy = round(df["y/n"].astype(int).sum()/df["y/n"].size*100, 2)
    print(f"{accuracy}% der Fragen wurden richtig beantwortet.")


# create dataframe file
eval_dict = {"input": ["Wie groß ist Max?", "Wie heißen die Kinder von Padme?", "Was braucht man für die Abschlussprüfung?"], "answer": ["Max ist 1,83 Meter groß", "Die Kinder von Padme heißen Luke und Leia.", "Zur Abschlussprüfung sollte man eine PowerPoint haben und zeigen wie der Prototyp funktioniert."]}
eval_df = pd.DataFrame(eval_dict)

# eval_df.to_json('Data/evaluation_data.json', orient="records",lines=True)

# evaluate the test questions        
evaluate(eval_df)

