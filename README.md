# Customer Support RAG Voice AI Agent

This project provides a voice-based customer support agent that uses local AI models and a Retrieval-Augmented Generation (RAG) architecture to answer queries based on a knowledge base of your documents. The agent is built using `fastrtc` for real-time communication and `langgraph` for creating a flexible, agentic workflow.

It now uses a fast, local `sentence-transformers` model for creating embeddings, ensuring a quick and completely private indexing process. It also automatically re-indexes your knowledge base whenever you add or modify documents.

## How it Works

The agent follows a Retrieval-Augmented Generation (RAG) pattern:

1.  **Speech-to-Text (ASR)**: The user's speech is converted to text.
2.  **Retrieve**: The agent searches a local knowledge base (a FAISS vector store created from your documents in the `documents/` directory) to find information relevant to the user's query. If no documents are found, this step is skipped.
3.  **Generate**: The retrieved information is used to augment a prompt, which is then passed to a local Large Language Model (`gemma:2b` via Ollama) to generate a response.
4.  **Text-to-Speech (TTS)**: The generated text is converted back into audio for the user to hear.

## How to Run

You can run the agent in two modes:

1.  **Gradio UI (Default)**: Launches a web interface for interaction.
    ```bash
    poetry run run-agent
    ```

2.  **Phone Interface**: Connects the agent to a temporary phone number for voice calls.
    ```bash
    poetry run run-agent --phone
    ```

### Efficient Indexing

The agent features an intelligent indexing system. On the first run, it builds a new knowledge base from your files. On subsequent runs, it automatically detects new or modified documents and **incrementally updates** the existing index. This ensures a fast startup time by avoiding unnecessary re-indexing of the entire knowledge base.

If you need to rebuild the index from scratch (e.g., if the index is corrupted or you've removed documents), you can use the `--reindex` flag:
```bash
poetry run run-agent --reindex
```

## Project Structure

The project is organized as follows:

-   `src/agent/`: Contains the core agent logic.
    -   `conversation_agent.py`: Defines the agent's graph and business logic.
    -   `main.py`: The entry point for running the agent.
    -   `vector_store.py`: Handles the creation and management of the FAISS index.
-   `documents/`: Contains the knowledge base for the agent. Add your files here.
-   `faiss_index/`: Stores the cached FAISS index for faster startup.
-   `prompts/`: Contains the prompt templates.
-   `pyproject.toml`: Defines the project dependencies and scripts.

## Customizing the Knowledge Base

To add your own information for the agent to use, simply add your PDF files to the `documents/` directory. The agent will automatically process them and update its knowledge base the next time you run it.

If the `documents/` directory is empty, the agent will still run but will operate in a conversational mode without a knowledge base. A warning will be logged to the console.
