import streamlit as st
from index import ask_question

st.set_page_config(
    page_title="Machine Learning Chatbot",
    page_icon="🤖"
)

st.title("🤖 Machine Learning Chatbot")

st.write("Ask questions from the Machine Learning PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


query = st.chat_input("Ask your question...")


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask_question(query)

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })