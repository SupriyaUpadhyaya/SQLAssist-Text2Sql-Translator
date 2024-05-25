import time
import torch
from utils import utils
import streamlit as st
from langchain.memory import ChatMessageHistory
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from core.refiner import Refiner
from core.rephraser import Rephraser
from core.context_retriever import ContextRetriever

st.title("🦙SQLAssist: NL2SQL Chatbot🤖")
st.markdown('#') 
st.sidebar.title("Settings")

if st.sidebar.checkbox("Follow up"):
    st.session_state.follow_up = True
else:
    st.session_state.follow_up = False
    
if st.sidebar.checkbox("Clear All"):
    st.session_state.messages = []
    st.session_state.query = ""
    st.session_state.history = ChatMessageHistory()

if "model" not in st.session_state:
    st.session_state["model"] = "basavaraj/text2sql-Llama3-8b"
    utils.load_model()
    st.session_state["db_path"] = "db/worlddb.db"

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    print("Creating session state")
        
st.session_state.topk = st.sidebar.slider("Use top k tables", 1, 3, 3)

# Display chat messages from history on app rerun
if 'messages' in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.15)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = utils.transcribe(prompt)
            _LOREM_IPSUM = response
            st.write_stream(stream_data)
    st.session_state.messages.append({"role": "assistant", "content": response})
   
if "query" in st.session_state and st.session_state.query != "":
    if st.toggle("View Query"):
        st.info(str(st.session_state.query))
    if st.toggle("View Logs"):
        st.info(str(st.session_state.current_log))
