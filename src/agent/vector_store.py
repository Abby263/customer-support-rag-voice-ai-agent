import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from loguru import logger

from .config import INDEX_PATH


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
    index_path_str = str(INDEX_PATH)
    if force_reindex and os.path.exists(index_path_str):
        logger.info(f"Force re-index requested. Deleting old index at '{index_path_str}'...")
        shutil.rmtree(index_path_str)

    if os.path.exists(index_path_str):
        logger.info(f"Loading existing FAISS index from '{index_path_str}'...")
        vector_store = FAISS.load_local(index_path_str, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
        logger.info("--- Vector DB Status ---")
        logger.info(f"  Vectors Loaded: {vector_store.index.ntotal}")
        logger.info(f"  Source: '{index_path_str}'")
        logger.info("------------------------")
    else:
        logger.info(
            f"No existing index found. Creating new index from documents in '{documents_path}'."
        )
        if not os.path.exists(documents_path) or not os.listdir(documents_path):
            logger.warning(
                f"The directory '{documents_path}' is empty or does not exist. "
                "The agent will run without a knowledge base. Add documents and restart to enable RAG."
            )
            return None

        logger.info("Loading documents...")
        loader = DirectoryLoader(path=str(documents_path), glob="**/*.pdf")
        docs = loader.load()
        if not docs:
            logger.warning(
                f"No documents found in '{documents_path}'. The agent will run without a knowledge base."
            )
            return None
        logger.info(f"Loaded {len(docs)} document(s).")

        logger.info("Splitting documents into chunks...")
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(split_docs)} chunks.")

        logger.info("Creating and embedding vector store... This may take a moment.")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        logger.info("Vector store created successfully.")

        logger.info(f"Saving new FAISS index to '{index_path_str}'.")
        vector_store.save_local(index_path_str)
        logger.info("Index saved successfully.")

        logger.info("--- Vector DB Indexing Complete ---")
        logger.info(f"  Documents Loaded: {len(docs)}")
        logger.info(f"  Content Chunks Created: {len(split_docs)}")
        logger.info(f"  Index saved to: '{index_path_str}'")
        logger.info("-----------------------------------")

    return vector_store


def update_vector_store(vector_store, embeddings, text_splitter, documents_path):
    """
    Updates an existing FAISS vector store with new or modified documents.
    """
    logger.info("Checking for new documents to update the index...")
    doc_path = Path(documents_path)
    
    # Get modification time of the index
    index_mtime = INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0

    # Find new or updated documents
    updated_files = [
        f for f in doc_path.rglob('**/*.pdf') 
        if f.is_file() and f.stat().st_mtime > index_mtime
    ]

    if not updated_files:
        logger.info("No new or modified documents found. Index is up to date.")
        return vector_store

    logger.info(f"Found {len(updated_files)} new or modified documents to add to the index.")
    
    # Load new documents
    docs = []
    for file_path in updated_files:
        loader = DirectoryLoader(path=str(file_path.parent), glob=f"**/{file_path.name}")
        docs.extend(loader.load())

    if not docs:
        return vector_store

    logger.info("Splitting new documents into chunks...")
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"Adding {len(split_docs)} new chunks to the vector store.")

    # Add new documents to the existing vector store
    vector_store.add_documents(split_docs)
    
    logger.info("Saving updated FAISS index...")
    vector_store.save_local(str(INDEX_PATH))
    logger.info("Index updated and saved successfully.")
    
    return vector_store 