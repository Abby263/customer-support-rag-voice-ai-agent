import os
import shutil
from typing import List, TypedDict
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from loguru import logger

from .vector_store import get_vector_store


class AgentState(TypedDict):
    question: str
    context: List[str]
    response: str


class CustomerCareAgent:
    def __init__(self, model_name="gemma3:4b", documents_path="documents", reindex=False):
        self.model = ChatOllama(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Use a faster, local embedding model
        logger.info("Initializing HuggingFaceEmbeddings with 'all-MiniLM-L6-v2'.")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Explicitly use CPU
        )

        # Define project root and prompt path
        project_root = Path(__file__).resolve().parent.parent.parent
        prompt_path = project_root / "prompts" / "customer_care.txt"

        # Load prompt from file
        try:
            with open(prompt_path, "r") as f:
                prompt_template_str = f.read()
            self.prompt = ChatPromptTemplate.from_template(prompt_template_str)
            logger.info(f"Prompt loaded from {prompt_path}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_path}")
            raise

        self.vector_store = get_vector_store(
            embeddings=self.embedding_model,
            text_splitter=self.text_splitter,
            documents_path=documents_path,
            force_reindex=reindex,
        )
        self.retriever = self.vector_store.as_retriever()
        self.graph = self.build_graph()

    def retrieve(self, state: AgentState):
        retrieved_docs = self.retriever.invoke(state["question"])
        return {"context": [doc.page_content for doc in retrieved_docs]}

    def generate(self, state: AgentState):
        chain = self.prompt | self.model
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        return {"response": response.content}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()


def create_agent(documents_path, reindex=False):
    agent_instance = CustomerCareAgent(documents_path=documents_path, reindex=reindex)
    return agent_instance.graph 