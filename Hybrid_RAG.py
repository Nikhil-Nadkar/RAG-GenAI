from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient, models
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


doc_file_path = "./data/Diet_Chart_PDF.pdf"
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
chunks = text_splitter.split_documents(documents=docs)

if not qdr.collection_exists(COLLECTION_NAME):
    qdr.create_collection(
        collection_name=COLLECTION_NAME, 
        vectors_config={"dense" : models.VectorParams(size=DIM, distance=models.Distance.COSINE)}, 
        sparse_vectors_config={"bm25" : models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    
    
texts = []
for chunk in chunks:
    texts.append(chunk.page_content)

vectors = emdeddings.embed_documents(texts=texts)

points = []
for i, vector in enumerate(vectors):
    payload = {**chunks[i].metadata, "text" : texts[i]}
    point = models.PointStruct(
        id = i,
        vector={
            "dense" : vector,
            "bm25" : models.Document(text=texts[i], model="Qdrant/bm25")
        },
        payload=payload
    )
    points.append(point)


qdr.upsert(collection_name=COLLECTION_NAME, points=points)

def Hybird_Search(query, k=5):
    q_vector = emdeddings.embed_query(query)
    
    records = qdr.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=q_vector,
                using="dense",
                limit=10
            ),
            
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="bm25",
                limit=10
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit = 5,
        with_payload=True
    )
    
    return records.points


query = "what are the key features of AI BASED CALORIE SCAN & DIET PLAN ?"

chunks = Hybird_Search(query)

context = ""
for chunk in chunks:
    context += chunk.payload.get("text") + "\n"
    

prompt = f"""
    You are a smart RAG system. You always provide the answer for user question based on the provided context.
    Qustion:{query}
    
    Context: {context}
"""

res = llm.invoke(prompt)

print(res.content)

