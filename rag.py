import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser() 

template = """
Beantworte die Frage basierend auf dem unten gegebenen Kontext.  

Kontext: {context}

Frage: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model | parser 

def generate_response(message, history):
    
    response = chain.invoke({
        "context": history, 
        "question": message
        })  
    return response
