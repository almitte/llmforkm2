import streamlit as st
import streamlit_authenticator as stauth
import rag
from confluence_api import get_data_confluence
import yaml
from vectorstore_functions import delete_vecs_pinecone, upsert_data_to_pinecone
from streamlit_option_menu import option_menu

# streamlit run ui.py


if "impl" not in st.session_state:
    st.session_state.impl = ""

with st.sidebar:
    selected = option_menu (
    menu_title = "Art des Retrievels",
    options = ["Basic", "History", "Sentence to Window", "HyDE", "Consecutive Subquestions", "Individual Subquestions", "Multiquery"]
    )

    match selected:
        case "Basic": st.session_state.impl = "basic"
        case "History": st.session_state.impl = "history"
        case "Sentence to Window": st.session_state.impl = "window"
        case "HyDE": st.session_state.impl = "hyde"
        case "Consecutive Subquestions": st.session_state.impl = "consecutive_subquestions"
        case "Individual Subquestions": st.session_state.impl = "individual_subquestions"
        case "Multiquery": st.session_state.impl = "multi"

st.title("Knowledge Management Chat")
st.write("Unsere Anwendung ermöglicht es dir, Confluence-Seiten in deinem Space automatisch zu extrahieren und dann mithilfe eines Language-Modeling-Modells (LLM) Fragen beantworten zu lassen. Darüber hinaus liefert sie relevante Seiten als Links zurück.")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] 

c1, c2 = st.columns(2)
clear = c1.button("Clear")
update = c2.button("Update")

# update button loads data from confluence
if update: 
    with st.status("Es dauert noch einen kleinen Moment"):    
    # updates Knowledge Base 
        # update json file
        get_data_confluence()
        st.write("Wissen wurde von Confluence extrahiert.")
        st.write("Wissen wird auf Pinecone hochgeladen...")
        # delete all existing vectors in pinecone vectorstore
        delete_vecs_pinecone()
        upsert_data_to_pinecone()
        st.write("Knowledge Base geupdated!")

# clear button clears all messages
if clear: 
    st.session_state.messages = []
    st.session_state.prompt_= False


# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])       

# session variable for prompt because it would go to false if the feedback buttons are clicked and then they wouldnt be reachible
if "prompt_" not in st.session_state:
    st.session_state.prompt_ =  False 

# function that changes session_state         
def callback():
    st.session_state.prompt_ =  True 

# speichern ther id der aktuellen Antwort    
if "run_id" not in st.session_state:
    st.session_state.run_id = ""

# react to user input
prompt = st.chat_input("Stellen Sie eine Frage", on_submit=callback)

if st.session_state.prompt_:
    if prompt != None:
        
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    # display assistant response in chat message container
    if st.session_state.messages[-1]["role"]!="assistant":
        # Antwort auf Basis des Verlaufs und des Prompts generieren
        with st.chat_message("assistant"):
            # tracing of run_id
            from langchain.callbacks import collect_runs
            with collect_runs() as cb:
                # not the entire history 
                history = st.session_state.messages[-7:-1]   
                # stream the input from generatorobject
                response = st.write_stream(rag.generate_response(prompt, history, st.session_state.impl))
                st.session_state.run_id=cb.traced_runs[0].id
                        
        # add assistant response to chat history  
        st.session_state.messages.append({"role": "assistant", "content": response})   
    col1, col2, col3 = st.columns([0.08,0.08,0.84])
    # feedback buttons
    if col1.button("👍"): 
        rag.send_feedback(st.session_state.run_id,1)
    
        st.rerun()
    if col2.button("👎"):
        rag.send_feedback(st.session_state.run_id,0) 
        
        st.rerun()
        
    # display links to the documents used
    with col3.expander("Links:"):
        if st.session_state.messages[-1]["content"]=="Ich weiß es nicht.":
            st.write("Keine relevanten Seiten gefunden!")
        else: 
            for source_tupel in rag.get_relevant_sources():
                title, source = source_tupel
                st.write(f"{title}: "  +  source)
                                            
    
            



            




