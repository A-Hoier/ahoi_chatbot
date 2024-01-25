from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
import openai
from langchain.chat_models import AzureChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

openai.api_type = os.environ["AZURE_OPENAI_API_TYPE"]
openai.base_url = ""

openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_version = os.environ["OPENAI_API_VERSION"]

openai.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]


def template(question, chat_history):
    return f"You are a helpful assistant. Here is your question:<{question}>, here is the chat history: <{chat_history}>"


def llm(model):
    print("inside llm function")
    return AzureChatOpenAI(
        azure_deployment=model,
        temperature=0,
        openai_api_key=openai.api_key,
        streaming=True,
    )


def chain() -> LLMChain:
    """This is DocBots chain."""

    # Use default template if a template is not passed to the chain
    qa_prompt = PromptTemplate.from_template(template())

    qa = LLMChain(
        llm=llm,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )
    return qa
