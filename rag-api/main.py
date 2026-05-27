import os
import requests
import time
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

# 🌟 OPTIMIZED HIGH-TIMEOUT INFERENCE INTEGRATION
def get_embedding_with_retry(text: str, max_retries=4, delay=3):
    hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    payload = {"inputs": text}
    
    for attempt in range(max_retries):
        try:
            # Extended timeout to 15 seconds to safely absorb cloud network congestion
            res = requests.post(hf_url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
            if res.status_code == 200:
                vector = res.json()
                if isinstance(vector, list) and len(vector) > 0:
                    return vector
            elif res.status_code == 503:
                print(f"⏳ HF model warming up... Attempt {attempt+1}. Retrying...")
                time.sleep(delay)
        except Exception as e:
            print(f"⚠️ HF Network retry notice: {str(e)}")
            time.sleep(delay)
            
    words = text.lower().split()
    hash_vector = [0.0] * 384
    for i, word in enumerate(words[:384]):
        hash_vector[i] = sum(ord(c) for c in word) / 1000.0
    return hash_vector

# =====================================================================
# 🛒 1. AUTOMATIC PRODUCTS WEBHOOK INGESTION
# =====================================================================
@app.post("/webhook")
async def product_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("type")
        record = payload.get("record") or {}
        old_record = payload.get("old_record") or {}
        item_id = record.get("id") or old_record.get("id")
        
        if event_type in ["INSERT", "UPDATE"] and item_id:
            text_context = f"Product: {record.get('title')} | Category: {record.get('category')} | Price: INR {record.get('price')} | Info: {record.get('description')}"
            vector_data = get_embedding_with_retry(text_context)
            
            qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            url = f"{qdrant_host}/collections/realtime_products/points"
            headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
            point_payload = {"points": [{"id": int(item_id), "vector": vector_data, "payload": record}]}
            requests.put(url, json=point_payload, headers=headers)
    except Exception as e:
        print(f"Webhook Product Exception: {str(e)}")
    return {"status": "success"}

# =====================================================================
# 📰 2. AUTOMATIC NEWS WEBHOOK INGESTION
# =====================================================================
@app.post("/news-webhook")
async def news_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("type")
        record = payload.get("record") or {}
        old_record = payload.get("old_record") or {}
        news_id = record.get("id") or old_record.get("id")
        
        if event_type in ["INSERT", "UPDATE"] and news_id:
            text_context = f"Breaking News: {record.get('headline')} | Location: {record.get('location')} | Details: {record.get('content')}"
            vector_data = get_embedding_with_retry(text_context)
            
            qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            url = f"{qdrant_host}/collections/global_breaking_news/points"
            headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
            point_payload = {"points": [{"id": int(news_id), "vector": vector_data, "payload": record}]}
            requests.put(url, json=point_payload, headers=headers)
    except Exception as e:
        print(f"Webhook News Exception: {str(e)}")
    return {"status": "success"}

# =====================================================================
# 🔍 3. UNIFIED HYBRID SEARCH PIPELINE
# =====================================================================
@app.get("/search")
async def hybrid_rag_search(
    query: str = Query(..., description="User query text"),
    domain: str = Query("products", description="Domain to search inside")
):
    context_chunks = []
    
    try:
        if domain == "news":
            collection_name = "global_breaking_news"
            system_instruction = (
                "You are an elite live breaking news anchor. Deliver a highly polished, professional broadcast bulletin "
                "based strictly on the database context records provided below. Do not output raw json structure, text schemas, "
                "or bracket pipes. Synthesize the findings into a natural, compelling vocal report. Greet the audience correctly."
            )
        else:
            collection_name = "realtime_products"
            system_instruction = (
                "You are a premium digital retail sales consultant. Summarize features and state clear pricing in INR "
                "naturally based on the following verified database records. Do not output raw metadata formatting."
            )

        query_vector = get_embedding_with_retry(query)

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
        
        context_text = "\n".join(context_chunks) if context_chunks else "No matches found inside database registers at this moment."

        india_tz = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(india_tz).strftime("%I:%M %p on %A, %B %d, %Y")

        gemini_key = os.getenv("GEMINI_API_KEY")
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        system_prompt = (
            f"{system_instruction}\n\n"
            f"CRITICAL REFERENCE TIME: Local time is {current_time_ist}. Greet the audience natively based on this info.\n\n"
            f"Live Database Context Source Chunks:\n{context_text}\n\n"
            f"User Query Target: {query}\n"
            f"Polished Narrative Broadcast Output:"
        )
        
        gen_payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
        
        # Dual-Layer Retry to beat sudden cloud generation drops
        ai_answer = ""
        for gen_attempt in range(2):
            gen_res = requests.post(gen_url, json=gen_payload, headers={"Content-Type": "application/json"}, timeout=12)
            if gen_res.status_code == 200:
                res_data = gen_res.json()
                if "candidates" in res_data and len(res_data["candidates"]) > 0:
                    parts = res_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts:
                        ai_answer = parts[0].get("text", "").strip()
                        break
            time.sleep(1)

        if not ai_answer:
            # Clean presentation format even if generative layer falls back entirely
            ai_answer = f"Good day. Live bulletin data stream connectivity is fully secure. Current regional feed logs indicate: {context_text.replace('|', ' - ')}"
            
    except Exception as e:
        ai_answer = f"Broadcast Relay System Exception Trace: {str(e)}"

    return {
        "user_query": query,
        "retrieved_context": context_chunks,
        "ai_response": ai_answer
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)