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

from .vector_store import get_vector_store, update_vector_store, INDEX_PATH


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
            model_kwargs={'device': 'cpu'}  # Explicitly use CPU
        )

        # Define project root and prompt path
        project_root = Path(__file__).resolve().parent.parent.parent
        prompt_path = project_root / "prompts" / "customer_care.txt"
        self.documents_path = project_root / documents_path

        # Load prompt from file
        try:
            with open(prompt_path, "r") as f:
                prompt_template_str = f.read()
            self.prompt = ChatPromptTemplate.from_template(prompt_template_str)
            logger.info(f"Prompt loaded from {prompt_path}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_path}")
            raise

        self.vector_store = self._get_or_create_vector_store(reindex)
        self.retriever = self.vector_store.as_retriever() if self.vector_store else None
        self.graph = self.build_graph()

    def _get_or_create_vector_store(self, force_reindex):
        """Gets or creates the vector store, handling automatic re-indexing."""
        index_path = Path(INDEX_PATH)
        documents_path = str(self.documents_path)

        if force_reindex and index_path.exists():
            logger.info("Force re-index requested. Deleting old index.")
            shutil.rmtree(index_path)

        if index_path.exists():
            logger.info(f"Loading existing FAISS index from '{index_path}'...")
            vector_store = FAISS.load_local(
                index_path, self.embedding_model, allow_dangerous_deserialization=True
            )
            # Check for updates
            vector_store = update_vector_store(
                vector_store, self.embedding_model, self.text_splitter, documents_path
            )
        else:
            vector_store = get_vector_store(
                embeddings=self.embedding_model,
                text_splitter=self.text_splitter,
                documents_path=documents_path,
                force_reindex=False, # Creation is handled here, so no need to force
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
            logger.info(f"    Content: {doc.page_content[:150]}...")  # Log first 150 chars

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


def create_agent(documents_path, reindex=False):
    agent_instance = CustomerCareAgent(documents_path=documents_path, reindex=reindex)
    return agent_instance.graph 