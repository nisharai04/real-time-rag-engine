import os
import requests
from fastapi import FastAPI, Query, Request
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Dynamically locate the .env file in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(title="Unified Production Search & Ingestion RAG Engine")

@app.get("/")
async def root_check():
    return {"status": "healthy", "message": "Unified Production Search Engine is active!"}

# =====================================================================
# 🛒 1. AUTOMATIC PRODUCTS WEBHOOK INGESTION (Directly inside Main App)
# =====================================================================
@app.post("/webhook")
async def product_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("type")
        record = payload.get("record", {})
        old_record = payload.get("old_record", {})
        item_id = record.get("id") or old_record.get("id")
        
        if event_type in ["INSERT", "UPDATE"]:
            text_context = f"Product: {record.get('title')} | Category: {record.get('category')} | Price: INR {record.get('price')} | Info: {record.get('description')}"
            
            # Generate 384 dimensional vector via HuggingFace
            hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
            hf_res = requests.post(hf_url, json={"inputs": text_context}, headers={"Content-Type": "application/json"})
            if hf_res.status_code == 200:
                vector_data = hf_res.json()
                
                # Push directly to Qdrant Cloud
                qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                url = f"{qdrant_host}/collections/realtime_products/points"
                headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
                point_payload = {
                    "points": [{
                        "id": int(item_id),
                        "vector": vector_data,
                        "payload": record
                    }]
                }
                requests.put(url, json=point_payload, headers=headers)
                print(f"🛒 Cloud-to-Cloud Product Vector Sync Successful for ID {item_id}!")
    except Exception as e:
        print(f"Webhook Product Ingestion Exception: {str(e)}")
    return {"status": "success"}

# =====================================================================
# 📰 2. AUTOMATIC NEWS WEBHOOK INGESTION (Directly inside Main App)
# =====================================================================
@app.post("/news-webhook")
async def news_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("type")
        record = payload.get("record", {})
        old_record = payload.get("old_record", {})
        news_id = record.get("id") or old_record.get("id")
        
        if event_type in ["INSERT", "UPDATE"]:
            text_context = f"Breaking News: {record.get('headline')} | Location: {record.get('location')} | Details: {record.get('content')}"
            
            # Generate 384 dimensional vector via HuggingFace
            hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
            hf_res = requests.post(hf_url, json={"inputs": text_context}, headers={"Content-Type": "application/json"})
            if hf_res.status_code == 200:
                vector_data = hf_res.json()
                
                # Push directly to Qdrant Cloud
                qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                url = f"{qdrant_host}/collections/global_breaking_news/points"
                headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
                point_payload = {
                    "points": [{
                        "id": int(news_id),
                        "vector": vector_data,
                        "payload": record
                    }]
                }
                requests.put(url, json=point_payload, headers=headers)
                print(f"📰 Cloud-to-Cloud News Vector Sync Successful for ID {news_id}!")
    except Exception as e:
        print(f"Webhook News Ingestion Exception: {str(e)}")
    return {"status": "success"}

# =====================================================================
# 🔍 3. UNIFIED HYBRID SEARCH PIPELINE
# =====================================================================
@app.get("/search")
async def hybrid_rag_search(
    query: str = Query(..., description="User query text"),
    domain: str = Query("products", description="Domain to search inside")
):
    print(f"🔍 Executing Bulletproof Search for '{domain}': '{query}'")
    context_chunks = []
    query_vector = None
    
    try:
        if domain == "news":
            collection_name = "global_breaking_news"
            system_instruction = (
                "You are a live, elite breaking news bulletin anchor. Summarize current regional events "
                "based strictly on the live database context records provided below. Speak in a neutral, clear, "
                "highly professional broadcasting tone. Greet the user naturally based on the exact time provided."
            )
        else:
            collection_name = "realtime_products"
            system_instruction = (
                "You are a friendly, expert retail store assistant. Talk naturally, mention prices clearly in INR, "
                "and summarize product specifications for the customer based on the live database context records. "
                "Greet the user naturally based on the exact time provided."
            )

        hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        hf_res = requests.post(hf_url, json={"inputs": query}, headers={"Content-Type": "application/json"})
        if hf_res.status_code == 200:
            query_vector = hf_res.json()

    except Exception as embed_err:
        print(f"⚠️ Vector Engine Error: {str(embed_err)}")

    if not query_vector or not isinstance(query_vector, list):
        words = query.lower().split()
        hash_vector = [0.0] * 384
        for i, word in enumerate(words[:384]):
            hash_vector[i] = sum(ord(c) for c in word) / 1000.0
        query_vector = hash_vector

    try:
        qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        url = f"{qdrant_host}/collections/{collection_name}/points/search"
        qdrant_headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
        qdrant_payload = {"vector": query_vector, "limit": 3, "with_payload": True}
        
        response = requests.post(url, json=qdrant_payload, headers=qdrant_headers)
        
        if response.status_code == 200:
            search_results = response.json().get("result", [])
            for res in search_results:
                p = res.get("payload")
                if p:
                    if domain == "news":
                        context_chunks.append(f"Headline: {p.get('headline')} | Location: {p.get('location')} | Update: {p.get('content')}")
                    else:
                        context_chunks.append(f"Product: {p.get('title')} | Category: {p.get('category')} | Price: INR {p.get('price')} | Info: {p.get('description')}")
        
        context_text = "\n".join(context_chunks) if context_chunks else "No dynamic cloud updates matches this exact intent right now."

        india_tz = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(india_tz).strftime("%I:%M %p on %A, %B %d, %Y")

        gemini_key = os.getenv("GEMINI_API_KEY")
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        system_prompt = (
            f"{system_instruction}\n\n"
            f"CRITICAL SYSTEM TIME INFO: Current local time for the user is {current_time_ist}. Greet accordingly.\n\n"
            f"Live Database Context Data:\n{context_text}\n\n"
            f"User Query: {query}\n"
            f"Answer:"
        )
        
        gen_payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
        gen_res = requests.post(gen_url, json=gen_payload, headers={"Content-Type": "application/json"})
        ai_answer = gen_res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            
    except Exception as e:
        ai_answer = f"Production System Handshake Exception: {str(e)}"

    return {
        "user_query": query,
        "retrieved_context": context_chunks,
        "ai_response": ai_answer
    }

iif __name__ == "__main__":
    import uvicorn
    # Render binds standard web services to port 10000 by default
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)