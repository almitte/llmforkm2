import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

def generate_response(prompt):
    parser = StrOutputParser() 
    chain = model | parser 
    return chain.invoke(prompt)