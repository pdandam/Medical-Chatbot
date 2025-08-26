from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # for chunking the documents
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data_path):
    """
    Load all PDF files from the specified directory and return a list of Document objects (text is extracted).
    """
    loader = DirectoryLoader(
        data_path, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Return new list of Document objects containing only source in metadata and the page content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split_docs(minimal_docs):
    """
    Split documents into chunks of text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20,
    )
    chunks = text_splitter.split_documents(minimal_docs)
    return chunks


def download_embeddings():
    """
    Download and return HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name)
    return embeddings