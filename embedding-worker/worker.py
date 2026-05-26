import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Dynamically locate the .env file in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(title="Multi-Domain Real-Time Ingestion Worker")

# Initialize Qdrant Client for cloud connection
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialize the local embedding model completely independent of Qdrant wrapper functions
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

PRODUCT_COLLECTION = "realtime_products"
NEWS_COLLECTION = "global_breaking_news"

# Auto-initialize Qdrant Collections if they don't exist
for col, size in [(PRODUCT_COLLECTION, 384), (NEWS_COLLECTION, 384)]:
    try:
        if not qdrant.collection_exists(col):
            qdrant.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )
            print(f"✨ Fresh Vector Space Verified/Created for '{col}'")
    except Exception as e:
        print(f"Collection status for {col}: {e}")

# 1. PRODUCTS ROUTE
@app.post("/webhook")
async def product_webhook(request: Request):
    payload = await request.json()
    event_type = payload.get("type")
    record = payload.get("record", {})
    old_record = payload.get("old_record", {})
    
    item_id = record.get("id") or old_record.get("id")
    
    if event_type in ["INSERT", "UPDATE"]:
        text_context = f"Product: {record.get('title')} | Category: {record.get('category')} | Price: INR {record.get('price')} | Info: {record.get('description')}"
        
        # Directly use standard encoder object
        vector_data = model.encode(text_context).tolist()
        
        qdrant.upload_points(
        collection_name=PRODUCT_COLLECTION,
        points=[
            PointStruct(
                id=int(item_id),
                vector=vector_data,
                payload=record
            )
        ]
        )
        print(f"🛒 Product Vector Sync Successful for ID {item_id}!")
        
    elif event_type == "DELETE":
        qdrant.delete(collection_name=PRODUCT_COLLECTION, points_selector=[int(item_id)])
        print(f"🗑️ Purged Product ID {item_id} vectors.")
        
    return {"status": "success"}

# 2. BREAKING NEWS ROUTE
@app.post("/news-webhook")
async def news_webhook(request: Request):
    payload = await request.json()
    event_type = payload.get("type")
    record = payload.get("record", {})
    old_record = payload.get("old_record", {})
    
    news_id = record.get("id") or old_record.get("id")
    
    if event_type in ["INSERT", "UPDATE"]:
        text_context = f"Breaking News: {record.get('headline')} | Location: {record.get('location')} | Details: {record.get('content')}"
        
        # Directly use standard encoder object
        vector_data = model.encode(text_context).tolist()
        
        qdrant.upload_points(
        collection_name=NEWS_COLLECTION,
        points=[
            PointStruct(
                id=int(news_id),
                vector=vector_data,
                payload=record
            )
        ]
    )
        print(f"📰 News Vector Sync Successful for ID {news_id}!")
        
    elif event_type == "DELETE":
        qdrant.delete(collection_name=NEWS_COLLECTION, points_selector=[int(news_id)])
        print(f"🗑️ Purged News ID {news_id} vectors.")
        
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)