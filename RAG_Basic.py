from dotenv import load_dotenv
load_dotenv()
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq


# load pdf/csv/exel/text/webpage
loader = PyMuPDFLoader(file_path="./data/Nikhil_Nadkar_Resume.pdf")
docs = loader.load()

# split loaded data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=100)
docs = text_splitter.split_documents(documents=docs)

# embedding model -- convert chunk data to numbers(embed)[2.3.5.0,134,-234]
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# inmemory vector store to store that embed chunks using embedded model you define
v_store = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)


llm  =  ChatGroq(
    model="openai/gpt-oss-20b"
)

query = "Tell Nikhil's total year of experience and Alembic Total years of experience ?"

# search similar chunks in vector store that matches query
chunks = v_store.similarity_search(query=query)

# add content of all similar chunks into one(bydefault we get 4 chunks for vector store but can control that by k=2)
content = ""

for chunk in chunks:
    content += chunk.page_content + "\n\n"

# prompt to give query and similar result from doc to generate output
prompt = f"""
    You are a smart RAG system. You always provide the answer for user question based on the provided content.
    Qustion:{query}
    
    Content: {content}
"""

res = llm.invoke(prompt)

print(res.content)
 

# print(len(query_results))
# print(query_results)
# print(len(docs))
# print(docs[1].page_content)
# print(docs[0].page_content)
# print(docs[0].metadata)