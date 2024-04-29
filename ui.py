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
    language = st.selectbox("Sprache:", ["deutsch", "englisch"])
    version = st.selectbox("Version:", ["gpt-3.5-turbo", "gpt-4","gpt-4-turbo"])

# Chatbot nur anzeigen wenn eingeloggt
if st.session_state["authentication_status"]:
    st.title("Knowledge Management Chat")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

    # Modellwechsel 
    if version:
        chain = rag.initialize_chain(version)

    col1, col2 = st.columns(2)
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
    
    if "buttons" not in st.session_state:
        st.session_state.buttons = [False, False]    

    # React to user input
    prompt = st.chat_input("Stellen Sie eine Frage", on_submit=callback)
    #st.markdown(prompt)
    #st.markdown(st.session_state.prompt_)
    if st.session_state.prompt_:
        if prompt != None:
            st.session_state.buttons = [False, False] # reset buttons
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        if st.session_state.messages[-1]["role"]!="assistant":
            # nur die letzten 10 Nachrichten beachten
        # history = st.session_state.messages[-10:]         
            # Antwort auf Basis des Verlaufs und des Prompts generieren
            with st.chat_message("assistant"):
                # Tracing der run_id
                from langchain.callbacks import collect_runs
                with collect_runs() as cb:
                    # Data wird gestreamt
                    response = st.write_stream(rag.generate_response(prompt, st.session_state.messages[-10:]   , chain))
                    # run-id zum feedback senden
                    st.session_state.run_id=cb.traced_runs[0].id
            # Add assistant response to chat history  
            st.session_state.messages.append({"role": "assistant", "content": response})            
        # feedback buttons
        col1, col2 = st.columns(2)
        if sum(st.session_state.buttons)==0:
            if col1.button("👍"): 
                rag.send_feedback(st.session_state.run_id,1)
                st.session_state.buttons[0] = True
                st.rerun()
            if col2.button("👎"):
                rag.send_feedback(st.session_state.run_id,0) 
                st.session_state.buttons[1] = True 
                st.rerun()
            
   


            




