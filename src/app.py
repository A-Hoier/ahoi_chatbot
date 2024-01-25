import os
from typing import List
import streamlit as st
import openai
from openai import AzureOpenAI



def run_app(self) -> None:  # noqa
    """The function that runs the application."""
    qa_chain = Query(st.session_state["chatbot"]).get_chain(template=(modified_template))
    else:
        qa_chain = Query(st.session_state["chatbot"]).get_chain(
            template=prompt_template
        )
    # Sidebar chatbot selection box.
    with st.sidebar:
        st.session_state["chatbot"] = st.selectbox(
            "Select the chatbot you want to use",
            st.session_state["chatbot_options"],
            index=st.session_state["chatbot_options"].index(
                st.session_state["chatbot"]
            ),
        )

    # Display intro message
    self.intro()

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
        if prompt := st.chat_input("Ask questions related to HR in Inspari "):
            # Reruns the vector database if the user requests
            # and a chatbot is currently selected
            if prompt == "rerun the vector db":
                if st.session_state["chatbot"] == "None":
                    with st.chat_message("assistant"):
                        st.write(
                            "Please pick a chatbot before reloading the vector db"
                        )
                else:
                    with st.chat_message("assistant"):
                        st.write(
                            "You've entered a secret command for"
                            " reinitializing the vector db "
                            f"for {st.session_state['chatbot']}"
                        )
                        st.write("Please stay on this page")
                        # Processing documents for reinitializing
                        # vector db for the selected chatbot
                        Ingest(
                            container_name=st.session_state["chatbot"]
                        ).process_documents("./data/")

                        # Display chat messages from the history
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    with st.chat_message("assistant"):
                        st.write("I've loaded the vector database for you. Enjoy!")

            else:
                # Limiting the history to the last 5 messages
                # in order to not use too much context
                if len(st.session_state["chat_history"]) > 6:
                    st.session_state["chat_history"] = st.session_state[
                        "chat_history"
                    ][-5:]
                # Handle user chat input, add user message to chat history
                st.session_state.messages.append(
                    {"role": "user", "content": prompt}
                )
                #   Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    # Request a response from the QnA chain. The response
                    # will also contain related documents.
                    response = qa_chain(
                        {
                            "question": prompt,
                            "chat_history": st.session_state["chat_history"],
                        }
                    )

                    # Listing the source documents of the responses.
                    sources = []
                    for source in response["source_documents"]:
                        document = os.path.basename(source.metadata["source"])
                        try:
                            page = source.metadata["page"] + 1
                            # page = int(page) + 1
                            final_source = f"{document},    page: {str(page)}"
                        except KeyError:
                            final_source = f"{document}"
                        if final_source not in sources:
                            sources.append(final_source)

                    for r in response["answer"]:
                        full_response += r
                        message_placeholder.markdown(full_response + "▌")
                        # time.sleep(random.uniform(0.001, 0.05))
                    message_placeholder.markdown(full_response)

                    k = "Sources: \n"
                    for s in sources:
                        k += "-  " + s + "\n"
                    st.markdown(k)
                    # Add assistant's messages to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    # Add sources to assistants content, so
                    # they will get printed on app rerun
                    st.session_state.messages.append(
                        {"role": "assistant", "content": k}
                    )



if __name__ == "__main__":
    run_app()
