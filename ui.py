import streamlit as st
import streamlit_authenticator as stauth
import rag
import data
import yaml
from yaml.loader import SafeLoader
# streamlit run interface.py



with st.sidebar:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    )
    authenticator.login()

    if st.session_state["authentication_status"]:
        authenticator.logout()
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

side1, side2 = st.columns(2)

# Chatbot nur anzeigen wenn eingeloggt
if st.session_state["authentication_status"]:
    st.title("Knowledge Management Chat")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

    c1, c2 = st.columns(2)
    clear = c1.button("Clear")
    upgrade = c2.button("Update")

    if upgrade: 
        with st.status("Es dauert noch einen kleinen Moment"):    
        # update die Knowledge Base 
            rag.update_data()
            st.write("Daten wurden erfolgreich geupdated")

    if clear: 
        st.session_state.messages = []
        st.session_state.prompt_= False

    # with st.chat_message("assistant"):
    #        st.write("Hallo, wie kann ich behilflich sein?")     

    # Display chat messages from history on app rerun
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
    
      

    # React to user input
    prompt = st.chat_input("Stellen Sie eine Frage", on_submit=callback)
    #st.markdown(prompt)
    #st.markdown(st.session_state.prompt_)
    if st.session_state.prompt_:
        if prompt != None:
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        if st.session_state.messages[-1]["role"]!="assistant":
            # Antwort auf Basis des Verlaufs und des Prompts generieren
            with st.chat_message("assistant"):
                # Tracing der run_id
                from langchain.callbacks import collect_runs
                with collect_runs() as cb:
                    # nur die letzten 10 Nachrichten beachten
                    history = st.session_state.messages[-10:-1]   
                    # Data wird gestreamt
                    response = st.write_stream(rag.generate_response(prompt, history, rag.initialize_chain()))
                    # run-id zum feedback senden
                    st.session_state.run_id=cb.traced_runs[0].id
                           
            # Add assistant response to chat history  
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
            st.markdown("Title1: https://docs.streamlit.io/develop/api-reference/layout/st.expander")
            st.markdown("TitleN: linkN")     
   


            




