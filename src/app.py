import os
from typing import List
import streamlit as st
import openai
from langchain.chat_models import AzureChatOpenAI
from logic.llmchain import chain


def run_app(self) -> None:  # noqa
    """The function that runs the application."""
    chain = chain()

    # Initialize chat history if it doesn't exist already
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Adding the chat_history in a format that is optimized for the LLM
    try:
        current_question = None
        for entry in st.session_state.messages:
            if entry["role"] == "user":
                current_question = entry["content"]
            elif entry["role"] == "assistant" and current_question:
                st.session_state["chat_history"].append(
                    (current_question, entry["content"])
                )
                current_question = None
    except AttributeError:
        pass

    # If no chatbot is selected,
    if st.session_state["chatbot"] == "None":
        return
    else:
        # Initiates streamlits internal memory
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        # Prompt the user for a new query and processes it.
        if prompt := st.chat_input("Go on. Ask ahead!"):
            # Reruns the vector database if the user requests
            # and a chatbot is currently selected

            st.session_state.messages.append({"role": "user", "content": prompt})
            #   Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Request a response from the QnA chain. The response
                # will also contain related documents.
                response = chain(
                    {
                        "question": prompt,
                        "chat_history": st.session_state["chat_history"],
                    }
                )
                for r in response["answer"]:
                    full_response += r
                    message_placeholder.markdown(full_response + "▌")
                    # time.sleep(random.uniform(0.001, 0.05))
                message_placeholder.markdown(full_response)

                # Add assistant's messages to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )


if __name__ == "__main__":
    run_app()
