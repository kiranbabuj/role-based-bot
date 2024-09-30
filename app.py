import streamlit as st
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Initialize session state to track the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(role="system", content="You are a customer of a bank. Your role is to upskill and train the employees of the bank.")
    ]

# Streamlit UI for OpenAI API key input
st.title("Finance Assistant")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Choose the LLM model
    model_choice = st.selectbox("Select an LLM model", ["gpt-3.5-turbo", "gpt-4o-mini"])

    # Initialize the LLM
    st.write("Initializing LLM model...")
   
