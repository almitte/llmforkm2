import streamlit as st
import rag
import data
# streamlit run interface.py

st.title("Knowledge Management Chat")

lang = st.selectbox("Sprache:", ["deutsch", "englisch"])


col1, col2 = st.columns(2)
clear = col1.button("Clear")
upgrade = col2.button("Update")

if upgrade: 
    with st.status("Es dauert noch einen kleinen Moment"):    
    # function die lange dauert 
    # update die Knowledge Base 
        data.update_data()

if clear: 
    st.session_state.messages = []  

# Initialize chat history
if "messages" not in st.session_state:
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

    response = rag.generate_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 
      




