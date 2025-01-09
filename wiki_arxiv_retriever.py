import streamlit as st
from LangchainAgents.knowledge import wiki_arxiv_agent


# give the title to our bot
st.title("WikiArxiv Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.chat_message("assistant").markdown("Hi, I am your assistant. Please ask me questions regarding any topic.")

# Display chat messages from history if rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take in the user input
if prompt := st.chat_input("Say something"):

    # display user message in chat container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Now the chatbot response section
    response = wiki_arxiv_agent(prompt)

    # display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content":response})