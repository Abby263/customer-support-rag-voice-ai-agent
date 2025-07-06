import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from loguru import logger

INDEX_PATH = "faiss_index"


def get_vector_store(embeddings, text_splitter, documents_path, force_reindex=False):
    """
    Creates or loads a FAISS vector store.

    Args:
        embeddings: The embedding model to use.
        text_splitter: The text splitter to use.
        documents_path: The path to the documents directory.
        force_reindex: Whether to force re-indexing.

    Returns:
        A FAISS vector store instance.
    """
    if force_reindex and os.path.exists(INDEX_PATH):
        logger.info(f"Force re-index requested. Deleting old index at '{INDEX_PATH}'...")
        shutil.rmtree(INDEX_PATH)

    if os.path.exists(INDEX_PATH):
        logger.info(f"Loading existing FAISS index from '{INDEX_PATH}'...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
        logger.info("--- Vector DB Status ---")
        logger.info(f"  Vectors Loaded: {vector_store.index.ntotal}")
        logger.info(f"  Source: '{INDEX_PATH}'")
        logger.info("------------------------")
    else:
        logger.info(
            f"No existing index found. Creating new index from documents in '{documents_path}'."
        )
        if not os.path.exists(documents_path) or not os.listdir(documents_path):
            raise FileNotFoundError(
                f"The directory '{documents_path}' is empty or does not exist. Please add documents for the agent to learn from."
            )

        logger.info("Loading documents...")
        loader = DirectoryLoader(path=str(documents_path), glob="**/*.pdf")
        docs = loader.load()
        if not docs:
            raise ValueError(f"No documents found in the '{documents_path}' directory.")
        logger.info(f"Loaded {len(docs)} document(s).")

        logger.info("Splitting documents into chunks...")
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(split_docs)} chunks.")

        logger.info("Creating and embedding vector store... This may take a moment.")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        logger.info("Vector store created successfully.")

        logger.info(f"Saving new FAISS index to '{INDEX_PATH}'.")
        vector_store.save_local(INDEX_PATH)
        logger.info("Index saved successfully.")

        logger.info("--- Vector DB Indexing Complete ---")
        logger.info(f"  Documents Loaded: {len(docs)}")
        logger.info(f"  Content Chunks Created: {len(split_docs)}")
        logger.info(f"  Index saved to: '{INDEX_PATH}'")
        logger.info("-----------------------------------")

    return vector_store 