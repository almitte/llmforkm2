import streamlit as st
import rag
import data

# streamlit run interface.py

st.title("Knowledge Management Chat")

col1, col2 = st.columns(2)
language = col1.selectbox("Sprache:", ["deutsch", "englisch"])
version = col2.selectbox("Version:", ["gpt-3.5-turbo", "gpt-4","gpt-4-turbo"])


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] 
### Modelwechsel funktioniert nichts    
if version == "gpt-4-turbo":
    chain = rag.initialize_everything(version)

if version == "gpt-4":
    chain = rag.initialize_everything(version)

if version == "gpt-3.5-turbo":
    # model, parser und chain inistialisieren
    chain = rag.initialize_everything(version)


clear = col1.button("Clear")
upgrade = col2.button("Update")

if upgrade: 
    with st.status("Es dauert noch einen kleinen Moment"):    
    # function die lange dauert 
    # update die Knowledge Base 
        data.update_data()
        st.write("Daten wurden erfolgreich geupdated")

if clear: 
    st.session_state.messages = []  



with st.chat_message("assistant"):
        st.write("Hallo, wie kann ich behilflich sein?")     

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])       
        
# React to user input
prompt = st.chat_input("Stellen Sie eine Frage")
if prompt :
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    if st.session_state.messages[-1]["role"]!="assistant":
       
        with st.spinner("Ich muss nachdenken..."):
            # nur die letzten 6 Nachrichten beachten
            history = st.session_state.messages[-10:]         
            # Antwort auf Basis des Verlaufs und des Prompts generieren
            response = rag.generate_response(prompt, history, chain)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response}) 
            
            with st.chat_message("assistant"):
                st.markdown(response)
        




