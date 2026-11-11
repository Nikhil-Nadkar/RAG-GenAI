from dotenv import load_dotenv
load_dotenv()

import base64, os
import pymupdf
import cohere
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

# function to covert page(texts, images, tables) to base64 url
def pdf_to_images(file_path):
    if not file_path:
        return "Please Provide a valid file path..."
    
    docs = pymupdf.open(file_path)
    
    urls, pngs = [], []
    for page in docs:
        png = page.get_pixmap().tobytes("png")   #covert whole page to png
        b64 = base64.b64encode(png).decode()        #covert png to base64 then to str(text)
        url = f"data:image/png;base64,{b64}"            #base64 url
        
        urls.append(url)
        pngs.append(png)
        
    
    return urls, pngs

# extracting urls and pngs form the doc
page_urls, page_pngs = pdf_to_images("./data/mix_data_sample.pdf")


# custom embedding class to insert images
class CohereImageEmbeddings(Embeddings):
    # define model and cohere
    def __init__(self, model = "embed-v4.0"):
        self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        self.model = model

    # embed images
    def embed_documents(self, images = []):
        if not images:
            return []
        
        result = self.client.embed(
            model=self.model,
            images=images,
            input_type="search_document",
            embedding_types=["float"]
        )
        
        vectors = [list(v) for v in result.embeddings.float_]
        return vectors
    
    # embed text - query
    def embed_query(self, text):
        if not text:
            return []
        
        result = self.client.embed(
            model=self.model,
            texts=[text],
            input_type="search_query",
            embedding_types=["float"]
        )
        
        
        return list(result.embeddings.float_[0])
    

# calling custom embed class
embeddings = CohereImageEmbeddings()

# embedding all the pages
page_vectors = embeddings.embed_documents(page_urls)

# QdrantClient setup
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "test_vectorDB"
DIM = 1536 #dimension

if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )


# preparing vector for insertion in vectordb
points = []
for i, vector in enumerate(page_vectors):
    point = PointStruct(
        id = i,
        vector = vector,
        payload={"page" : i}
    )
    
    points.append(point)

# inserting into vector database
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

# user query
query = "Tell me about the invoice no. 2022435 ?"
v_query = embeddings.embed_query(query)


# Finding similarity_search from the vector qdrant
result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=v_query,
    limit=2
).points


# getting the topPageNO to get the image from page_urls 
topPageNo = []

for point in result:
    pageNo = point.payload.get("page")
    topPageNo.append(pageNo)

print(topPageNo)
img = page_urls[topPageNo[0]]


# llm call to get answer 
llm = ChatCohere(
    model="command-a-plus-05-2026"
)

prompt = HumanMessage(
    content=[
        {
            "type": "text",
            "text": f"Answer the question using the provided context image. Question: {query}"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": img
            }
        }
    ]
)

res = llm.invoke([prompt])

print(res.content[1].get("text"))
