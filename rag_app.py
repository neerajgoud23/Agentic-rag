from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.groq import Groq 
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

urls = [
    "https://qdrant.tech/documentation/overview/",
]

loader = WebBaseLoader(urls)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(loader.load())

embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")

client = QdrantClient(path="C:/Users/thanm/Desktop/Agentic AI")
collection_name = "test"

try:
    collection_info = client.get_collection(collection_name=collection_name)
except (UnexpectedResponse, ValueError):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)

vector_store.add_documents(documents=chunks)
retriever = vector_store.as_retriever()

knowledge_base = LangChainKnowledgeBase(retriever=retriever)


agent = Agent(
    model=Groq("deepseek-r1-distill-qwen-32b"),
    knowledge=knowledge_base,
    description="Answer to the user question from the knowledge base",
    markdown=True,
    search_knowledge=True,
)

user_query = "The most commonly used distance metrics as per the document"
agent.print_response(user_query, stream=True)