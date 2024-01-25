import os
from typing import List

import chromadb  # noqa
import langchain
import openai
import streamlit as st
#from azure.storage.blob import BlobServiceClient, ContainerClient
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from stqdm import stqdm

# __import__("pysqlite3")  # noqa
# import sys  # noqa

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # noqa
# import sqlite3  # noqa


class Ingest:
    """This class is used to create the vector store."""

    def __init__(self, container_name: str, persist_dir: str = "./docs/chroma") -> None:
        """Init function for the Ingest layer."""
        self.container_name = container_name
        self.persist_directory = "./data/"
        self.deployment_name = "gpt35turbo"
        # Connection string to the data lake
        self.connection_string = os.environ["docbotstorage"]
        self.container_client = ContainerClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=self.container_name,
        )
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.resource_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        # OpenAI settings
        openai.api_type = "azure"
        openai.api_base = self.resource_endpoint
        openai.api_key = self.api_key
        openai.api_version = "2023-05-15"
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = self.resource_endpoint
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        # The LLM chat model
        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            model_name="gpt-35-turbo",
            temperature=0,
            openai_api_base=openai.api_base,
            openai_api_key=openai.api_key,
        )
        # Embedding the data base entries and the prompt
        self.embedding_function = OpenAIEmbeddings(deployment="ada-embedding-hr")
        # Chroma DB client
        self.client = chromadb.PersistentClient(persist_dir)

    def list_containers(self) -> List[str]:
        """Used to list the containers in the data lake."""
        blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=self.connection_string, container_name=self.container_name
        )
        containers = blob_service_client.list_containers()
        return containers

    def _download_documents(self) -> None:
        """Downloads the document from the storage account."""
        print("downloading blobs")
        # Retrieves a list of all the blobs
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print(blob.name)
            blob_service_client = BlobServiceClient.from_connection_string(
                conn_str=self.connection_string, container_name=self.container_name
            )
            blob_client = blob_service_client.get_blob_client(
                container=self.container_name, blob=blob.name
            )
            # Creates a path for storing the data on disk
            data_path = "data/" + self.container_name

            if not os.path.exists(data_path):
                os.makedirs(data_path)
            final_path = os.path.join(data_path, blob.name)
            # Downloads the blob to the disk
            if not os.path.exists(final_path):
                with open(file=final_path, mode="wb") as sample_blob:
                    download_stream = blob_client.download_blob()
                    sample_blob.write(download_stream.readall())

    def _load_pdf(self, dir: str) -> List[str]:
        """Used for loading PDF files."""
        loader = DirectoryLoader(
            path=dir,
            glob="*.pdf",
            show_progress=True,
            use_multithreading=True,
            loader_cls=PyPDFLoader,
        )
        return loader.load()

    def _load_pptx(self, dir: str) -> List[str]:
        """Used for loading PPTX files."""
        loader = DirectoryLoader(
            path=dir,
            glob="*.pptx",
            show_progress=True,
            use_multithreading=True,
            loader_cls=UnstructuredPowerPointLoader,
        )
        return loader.load()

    def _split_text(self, documents: List[str]) -> List[str]:
        """Split the documents in chunks."""
        chunk_size = 600
        chunk_overlap = 50
        # if self.container_name == "hr-docs":
        #     chunk_size = 300
        #     chunk_overlap = 40
        # else:
        #     chunk_size = 500
        #     chunk_overlap = 50

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return text_splitter.split_documents(documents)

    def _add_documents(
        self, splits: List[langchain.schema.document.Document]
    ) -> chromadb.PersistentClient:
        """Adds the documents to the Chroma DB."""
        ef = self.embedding_function
        # Loads or creates the collection
        collection = self._get_collection()
        embeddings = []
        documents_list = []
        metadatas = []
        ids = []
        i = "1"
        # Iterates through the splits, and stores the necessary
        # information on each of them
        for s in stqdm(splits):
            embeddings.append(ef.embed_query(s.page_content.replace("\n", " ")))
            documents_list.append(s.page_content.replace("\n", " "))
            metadatas.append(s.metadata)
            ids.append("id" + i)
            i = str(int(i) + 1)

        collection.add(
            embeddings=embeddings,
            documents=documents_list,
            metadatas=metadatas,
            ids=ids,
        )

        return collection

    def _delete_collection(self) -> None:
        """Delete collection."""
        self.client.delete_collection(name=self.container_name)

    def _get_collection(self) -> chromadb.PersistentClient:
        """Create or load the collection."""
        coll = self.client.get_or_create_collection(name=self.container_name)
        return coll

    def get_vectorstore(self) -> Chroma:
        """Provides the vectorstore / vector DB."""
        vs = Chroma(
            client=self.client,
            collection_name=self.container_name,
            embedding_function=self.embedding_function,
        )
        return vs

    def process_documents(self, file_path: str) -> None:
        """Download documents from the blob storage.

        Load them into the vector DB.
        """
        # Download documents
        self._download_documents()
        file_path = file_path + self.container_name
        files = []
        for file in os.listdir(file_path):
            files.append(os.path.join(file_path, file))
        # Separates the different file formats
        pdf_files = [x for x in files if x.endswith(".pdf")]
        pptx_files = [x for x in files if x.endswith(".pptx")]
        docs = []
        # Extend the docs list with the loaded files
        if pdf_files:
            docs.extend(self._load_pdf(file_path))
            splits = self._split_text(docs)
        if pptx_files:
            print(pptx_files)
            docs.extend(self._load_pptx(file_path))
            splits = docs
        coll = self._get_collection()
        # Initializes the vector database if a database with
        # the same name and size doesn't already exist.
        if coll.count() != len(splits):
            self._delete_collection()
            coll = self._add_documents(splits)
        st.write(
            "The Ingest function has completed"
            "and the vector database now contains "
            f"{coll.count()} documents"
        )


class Query:
    """Used for querying the docs."""

    def __init__(self, container_name: str) -> None:
        """Init function for the query class."""
        self.deployment_name = "gpt35turbo"
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.resource_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.container_name = container_name
        openai.api_type = "azure"
        openai.api_base = self.resource_endpoint
        openai.api_key = self.api_key
        openai.api_version = "2023-05-15"
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = self.resource_endpoint
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        # The LLM chat model
        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            model_name="gpt-35-turbo",
            temperature=0,
            openai_api_base=openai.api_base,
            openai_api_key=openai.api_key,
            streaming=True,
        )

    def get_chain(self, template: str = "") -> ConversationalRetrievalChain:
        """This is DocBots chain."""
        # Creates the prompt tempalte from the template variable
        qa_prompt = PromptTemplate.from_template(template=template)
        # Use default template if a template is not passed to the chain
        if template == "":
            qa_prompt = PromptTemplate.from_template(
                template=TemplateLoader().load_template()
            )
        # This template is used to creating a new question for the
        # database query. The new question is created based upon
        # the chat history and the most recent question
        condensed_question_prompt = PromptTemplate.from_template(
            template=TemplateLoader().load_condensed_template()
        )
        # The number of documents to pass to the LLM
        # together with the question
        k = 4
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            condense_question_llm=self.llm,
            condense_question_prompt=condensed_question_prompt,
            retriever=Ingest(container_name=self.container_name)
            .get_vectorstore()
            .as_retriever(k=k),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True,
        )
        return qa


class App:
    """Web interface for the chatbot."""

    def __init__(self) -> None:
        """Init function for application layer."""
        #self.connection_string = os.environ["docbotstorage"]

    def get_container_list(self) -> List[str]:
        """Creates a list of chat models."""
        output = ["None"]

        # Exlude containers that are not related to a specific chatbot
        containers_to_exclude = [
            "azure-webjobs-hosts",
            "azure-webjobs-secrets",
            "scm-releases",
            "templates",
        ]

        # Add each relevant container to the list
        try:
            blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            all_containers = blob_service_client.list_containers()
            for container in all_containers:
                if container["name"] not in containers_to_exclude:
                    output.append(container["name"])
        except Exception as ex:
            print(f"Exception: {ex}")
        return output

    def intro(self) -> None:
        """The introduction on the front page."""
        if st.session_state["chatbot"] == "None":
            st.title("Please select af model in the sidebar")
        # Intro for the hr-docs Doc-Bot
        elif st.session_state["chatbot"] == "hr-docs":
            st.title(
                "Welcome to DocBot"
                "\n Query Insparis HR documents through natural text"
            )
            with st.chat_message("assistant"):
                st.write(
                    "Hello, I am Insparis chatbot, but you can call me DocBot."
                    "Ask me anything related to HR in Inspari,"
                    "and I'll be happy to help."
                )
        # Intro for the cvs CV-Bot
        elif st.session_state["chatbot"] == "cvs":
            st.title("Welcome to Insparis cv generator bot")
            with st.chat_message("assistant"):
                st.write(
                    "Hello, I'm Insparis CV-Bot."
                    "Paste a job posting and I'll match you up with"
                    " potential candidates"
                )
        # Intro for the Custom-Bot
        else:
            st.title("Welcome to your custom chat bot")
            with st.chat_message("assistant"):
                st.write(
                    "Hello, I'm your Custom-Bot. "
                    "You can start asking question about the related documents."
                )

    def run_app(self) -> None:  # noqa
        """The function that runs the application.

        This main function orchestrates the whole chatbot application.
        """
        # Fetch and store available chatbot container list
        st.session_state["chatbot_options"] = self.get_container_list()

        # Initializing chatbot state to "None" if it has not been assigned yet
        if "chatbot" not in st.session_state:
            st.session_state["chatbot"] = "None"

        # Initializing the prompt template state if it has not been assigned yet
        if "prompt_template" not in st.session_state:
            st.session_state["prompt_template"] = {}

        # Associate existing prompt template with the chosen chatbot.
        # If the prompt template does not exist, it loads a new one
        if st.session_state["chatbot"] in st.session_state["prompt_template"]:
            prompt_template = st.session_state["prompt_template"][
                st.session_state["chatbot"]
            ]
        else:
            prompt_template = TemplateLoader().load_template()
            st.session_state["prompt_template"][
                st.session_state["chatbot"]
            ] = prompt_template

        # Retrieves the modified template from the
        # current session and use it as the primary template
        if "new_prompt" in st.session_state:
            modified_template = st.session_state["new_prompt"]
            # Store the modified template for the chosen chatbot in the session state
            st.session_state["prompt_template"][
                st.session_state["chatbot"]
            ] = modified_template
            qa_chain = Query(st.session_state["chatbot"]).get_chain(
                template=(modified_template)
            )
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


# This class fetches, uploads and manipulates prompt templates from blob storage
class TemplateLoader:
    """Class for loading templates from Blob Storage."""

    def __init__(self) -> None:
        """Initialize blob service client using environment variables."""
        self.connection_string = os.environ["docbotstorage"]
        self.blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=self.connection_string
        )

    def _read_blob_data(self, blob_name: str) -> bytes:
        """Internally used function to fetch a blob's data using its name."""
        blob_client = self.blob_service_client.get_blob_client("templates", blob_name)
        # Download and read blob data
        blob_data = blob_client.download_blob().readall()
        return blob_data

    def load_template(self) -> str:
        """Fetches, decodes and loads a template from blob storage.

        If no chatbot is selected, it loads the default template.
        """
        key = st.session_state["chatbot"]
        if key == "None":
            key = "default"
        template_blob_data = self._read_blob_data(f"{key}/{key}.template")
        # Decodes the downloaded blob data
        template = template_blob_data.decode("utf-8")
        return template

    def load_condensed_template(self) -> str:
        """Fetches, decodes and loads a condensed template from blob storage.

        If no chatbot is selected, it loads the default condensed template.
        """
        key = st.session_state["chatbot"]
        if key == "None":
            key = "default"
        template_blob_data = self._read_blob_data(f"{key}/{key}-condensed.template")
        # Decodes the downloaded condensed blob data
        template = template_blob_data.decode("utf-8")
        # template = str(template_blob_data).replace("b", "").replace("'", "")
        return template

    def save_template(self, key: str, template: str) -> None:
        """Saves a template to blob storage under the given key.

        If the key is marked as "None", the template is saved under "default".
        """
        if key == "None":
            key = "default"
        blob_name = f"{key}/{key}.template"
        print(blob_name)
        container_client = self.blob_service_client.get_container_client(
            container="templates"
        )
        # Uploads a new template to the blob
        container_client.upload_blob(name=blob_name, data=template, overwrite=True)

    def save_condensing_template(self, key: str, template: str) -> None:
        """Saves a condensed template to blob storage under the given key.

        The condensed template structure is defined here.
        """
        blob_name = f"{key}/{key}-condensed.template"
        template = (
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question."
            " You can assume the question to be about: "
            f"{template}. "
            "Chat history and question are delimited by <>. "
            "chat history: <{chat_history}>. "
            "Follow Up Input: <{question}>. "
            "Standalone question: "
        )
        container_client = self.blob_service_client.get_container_client(
            container="templates"
        )
        # Uploads a new condensed template to the blob
        container_client.upload_blob(name=blob_name, data=template, overwrite=True)


if __name__ == "__main__":
    App().run_app()
