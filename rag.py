import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate 
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



def initialize_everything(name):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=name)
    parser = StrOutputParser() 

    # template = """
    # Beantworte die Frage basierend auf dem gegebenen Kontext: {context}
    
    
    # Wenn das Beantworten der Frage nicht möglich ist durch den gegebenen Kontekt, antworte "Ich weiß es nicht". 
    # """
    template="Beantworte die Frage"

    prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
    ])

    chain = prompt | model | parser 
    
    
    return chain 

def generate_response(message, history_list, chain):
    history =[]
    for mes in history_list:
        # verschiedene Wrapper für verschiedene roles
        m = HumanMessage(content=mes["content"]) if mes["role"] == "user" else AIMessage(content=mes["content"])
        history.append(m)

    response = chain.invoke({
        "history": history, 
        "question": message,
        "context": ""
        })  
    
    return response

message = "Wie heiße ich?"
history_list=[{"role":"user", "content": "Ich heiße Max"}, {"role": "assistant", "content": "Schön dich kennen zu lernen"}]

chain = initialize_everything("gpt-4")
print(generate_response(message, history_list, chain))