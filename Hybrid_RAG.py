from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


doc_file_path = "./data/mix_data_sample.pdf"
Chunk_Size = 500
Chunk_Overlap = 100
Embed_Model = "embed-v4.0"
LLM_Model = "openai/gpt-oss-20b"
qdr_url = "http://localhost:6333"
DIM = 1536
COLLECTION_NAME = "hybrid_rag"


qdr = QdrantClient(url=qdr_url)
emdeddings = CohereEmbeddings(model=Embed_Model, cohere_api_key=os.getenv("COHERE_API_KEY"))
llm = ChatGroq(model=LLM_Model)


loader = PyMuPDFLoader(file_path = doc_file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = Chunk_Size, chunk_overlap = Chunk_Overlap)
docs = text_splitter.split_documents(documents=docs)





print(len(docs))

