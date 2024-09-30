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
st.title("Learning Assistant")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Choose the LLM model
    model_choice = st.selectbox("Select an LLM model", ["gpt-3.5-turbo", "gpt-4o-mini"])

    # Initialize the LLM
    st.write("Initializing LLM model...")
    llm = OpenAI(model=model_choice)

    # Chat interaction using LLM
    st.write("Chat with the LLM")
    user_input = st.text_input("Ask your question:")

    if user_input:
        # Add user message to the history
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))

        try:
            # Query the LLM
            response = llm.chat(st.session_state.messages)
            if isinstance(response, list):
                # If the response is a list of messages, extract the last message's content
                assistant_response = response[-1].content if len(response) > 0 else "No response"
            else:
                # If response is not a list, assume it's a direct response
                assistant_response = str(response)

            # Add assistant response to the history
            st.session_state.messages.append(ChatMessage(role="customer", content=assistant_response))
        except Exception as e:
            st.error(f"Error while querying the LLM: {e}")

    # Display the conversation history
    st.write("## Conversation History")
    for msg in st.session_state.messages:
        if msg.role == "user":
            st.write(f"**You**: {msg.content}")
        elif msg.role == "customer":
            st.write(f"**Customer**: {msg.content}")
        else:
            st.write(f"**System**: {msg.content}")

    # Embeddings section
    st.write("## Embeddings Section")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", max_tokens=300)

    # Document loading and querying
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF locally
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load documents and create an index
        documents = SimpleDirectoryReader("./").load_data()
        index = VectorStoreIndex.from_documents(documents)

        # Query engine
        query = st.text_input("Ask something from the uploaded documents:")
        if query:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            st.write(f"Document Response: {response}")
else:
    st.warning("Please enter your OpenAI API key to continue.")
