import streamlit as st
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9,
)

# Streamlit page config
st.set_page_config(page_title="Funny Mistral Chatbot", page_icon="🤖")

st.title("🤖 Funny Mistral Chatbot")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a Funny assistant.")
    ]

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    # Get response from model
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.write(response.content)