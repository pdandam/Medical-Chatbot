from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split_docs, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

extracted_data = load_pdf_files("data/")
filter_data = filter_to_minimal_docs(extracted_data)
chunks = text_split_docs(filter_data)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pinecone_client = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pinecone_client.has_index(index_name):
    pinecone_client.create_index(
        name=index_name,
        dimension=384,  # dimension of the embedding model
        metric="cosine",  # distance metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pinecone_client.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    index_name=index_name
)