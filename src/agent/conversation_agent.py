import os
import shutil
from typing import List, TypedDict

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from loguru import logger

from .vector_store import get_vector_store, update_vector_store
from .config import (
    AGENT_CONFIG,
    TEXT_SPLITTER_CONFIG,
    PROMPT_PATH,
    DOCUMENTS_PATH,
    INDEX_PATH,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    response: str


class CustomerCareAgent:
    def __init__(self, reindex=False):
        self.model = ChatOllama(model=AGENT_CONFIG.get("model_name", "gemma:2b"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG.get("chunk_size", 1000),
            chunk_overlap=TEXT_SPLITTER_CONFIG.get("chunk_overlap", 200),
        )

        # Use a faster, local embedding model
        embedding_model_name = AGENT_CONFIG.get("embedding_model", "all-MiniLM-L6-v2")
        logger.info(f"Initializing HuggingFaceEmbeddings with '{embedding_model_name}'.")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}  # Explicitly use CPU
        )

        # Load prompt from file
        try:
            with open(PROMPT_PATH, "r") as f:
                prompt_template_str = f.read()
            self.prompt = ChatPromptTemplate.from_template(prompt_template_str)
            logger.info(f"Prompt loaded from {PROMPT_PATH}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {PROMPT_PATH}")
            raise

        self.vector_store = self._get_or_create_vector_store(reindex)
        self.retriever = self.vector_store.as_retriever() if self.vector_store else None
        self.graph = self.build_graph()

    def _get_or_create_vector_store(self, force_reindex):
        """Gets or creates the vector store, handling automatic re-indexing."""
        if force_reindex and INDEX_PATH.exists():
            logger.info("Force re-index requested. Deleting old index.")
            shutil.rmtree(INDEX_PATH)

        if INDEX_PATH.exists():
            logger.info(f"Loading existing FAISS index from '{INDEX_PATH}'...")
            vector_store = FAISS.load_local(
                str(INDEX_PATH), self.embedding_model, allow_dangerous_deserialization=True
            )
            # Check for updates
            vector_store = update_vector_store(
                vector_store, self.embedding_model, self.text_splitter, str(DOCUMENTS_PATH)
            )
        else:
            vector_store = get_vector_store(
                embeddings=self.embedding_model,
                text_splitter=self.text_splitter,
                documents_path=str(DOCUMENTS_PATH),
                force_reindex=False,  # Creation is handled here, so no need to force
            )

        return vector_store

    def retrieve(self, state: AgentState):
        if not self.retriever:
            return {"context": []}
        retrieved_docs = self.retriever.invoke(state["question"])
        logger.info(f"Retrieved {len(retrieved_docs)} document chunks for the question: '{state['question']}'")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"  Chunk {i+1}:")
            logger.info(f"    Source: {doc.metadata.get('source', 'Unknown')}")

        return {"context": [doc.page_content for doc in retrieved_docs]}

    def generate(self, state: AgentState):
        chain = self.prompt | self.model
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        return {"response": response.content}

    def generate_no_rag(self, state: AgentState):
        """Generates a response without RAG if no documents are available."""
        chain = self.prompt | self.model
        # Provide empty context
        response = chain.invoke({"context": [""], "question": state["question"]})
        return {"response": response.content}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        if self.retriever:
            workflow.add_node("retrieve", self.retrieve)
            workflow.add_node("generate", self.generate)
            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "generate")
        else:
            workflow.add_node("generate_no_rag", self.generate_no_rag)
            workflow.set_entry_point("generate_no_rag")
            workflow.add_edge("generate_no_rag", END)

        workflow.add_edge("generate", END)
        return workflow.compile()


def create_agent(reindex=False):
    agent_instance = CustomerCareAgent(reindex=reindex)
    return agent_instance.graph 