import os
import requests
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Dynamically locate the .env file in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(title="Hybrid Multi-Domain RAG Search Engine")

@app.get("/")
async def root_check():
    return {"status": "healthy", "message": "Production Search Engine is fully active!"}

@app.get("/search")
async def hybrid_rag_search(
    query: str = Query(..., description="User query text"),
    domain: str = Query("products", description="Domain to search inside")
):
    print(f"🔍 Executing Production Semantic Search for '{domain}': '{query}'")
    
    context_chunks = []
    
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

        # 1. 🌟 DIRECT PRODUCTION REST EMBEDDING CALL (Bypasses SDK blocks completely)
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Using the standard production text-embedding-004 route with clean query params
        embed_url = f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key={gemini_key}"
        embed_payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": query}]}
        }
        
        embed_res = requests.post(embed_url, json=embed_payload, headers={"Content-Type": "application/json"})
        
        if embed_res.status_code == 200:
            query_vector = embed_res.json()["embedding"]["values"]
        else:
            print(f"⚠️ Primary embed failed, running fallback endpoint: {embed_res.text}")
            # Global stable deployment path fallback
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={gemini_key}"
            embed_res = requests.post(fallback_url, json=embed_payload, headers={"Content-Type": "application/json"})
            query_vector = embed_res.json()["embedding"]["values"]

        # 2. Asli Semantic Match against Qdrant Cloud Vectors
        qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        url = f"{qdrant_host}/collections/{collection_name}/points/search"
        headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
        payload = {
            "vector": query_vector,
            "limit": 3,  # Scaled up context retrieval window
            "with_payload": True
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
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

        # 3. ⏰ TIMEZONE CORRECTION: Inject Indian Time dynamically into Gemini
        india_tz = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(india_tz).strftime("%I:%M %p on %A, %B %d, %Y")

        # 4. Final Generation REST Request
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        
        system_prompt = (
            f"{system_instruction}\n\n"
            f"CRITICAL SYSTEM TIME INFO: Current local time for the user is {current_time_ist}. Greet accordingly.\n\n"
            f"Live Database Context Data:\n{context_text}\n\n"
            f"User Query: {query}\n"
            f"Answer:"
        )
        
        gen_payload = {
            "contents": [{"parts": [{"text": system_prompt}]}]
        }
        
        gen_res = requests.post(gen_url, json=gen_payload, headers={"Content-Type": "application/json"})
        ai_answer = gen_res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            
    except Exception as e:
        ai_answer = f"Production System Handshake Exception: {str(e)}"

    return {
        "user_query": query,
        "retrieved_context": context_chunks,
        "ai_response": ai_answer
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)