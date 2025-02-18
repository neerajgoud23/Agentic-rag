# Agentic-rag


Agentic rag is a project that leverages the LangChain framework to create an intelligent agent capable of answering user questions based on a knowledge base. The project uses various components such as document loaders, text splitters, embeddings, and vector stores to build a robust retrieval-augmented generation (RAG) system.

## Features

- **Document Loading**: Load documents from web sources using `WebBaseLoader`.
- **Text Splitting**: Split documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Embeddings**: Generate embeddings for document chunks using `FastEmbedEmbeddings`.
- **Vector Store**: Store and retrieve document embeddings using `QdrantVectorStore`.
- **Knowledge Base**: Create a knowledge base using `LangChainKnowledgeBase`.
- **Intelligent Agent**: Use the `Agent` class to interact with the knowledge base and answer user queries.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentic-ai.git
   cd agentic-ai
   
2. Install the required packages:
    pip install -r requirements1.txt
   
3.Set up environment variables: Create a .env file in the root directory and add your GROQ_API_KEY:
    GROQ_API_KEY=your_groq_api_key


    
