import os
import requests
from fastapi import FastAPI, Query
from dotenv import load_dotenv

# Dynamically locate the .env file in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(title="Hybrid Multi-Domain RAG Search Engine")

@app.get("/")
async def root_check():
    return {"status": "healthy", "message": "Search Engine is running!"}

@app.get("/search")
async def hybrid_rag_search(
    query: str = Query(..., description="User query text"),
    domain: str = Query("products", description="Domain to search inside")
):
    print(f"🔍 Searching domain '{domain}' space for: '{query}'")
    
    context_chunks = []
    
    try:
        if domain == "news":
            collection_name = "global_breaking_news"
            system_instruction = (
                "You are a live breaking news assistant. Summarize regional events based on "
                "the live database context provided below. Speak in a clear, broadcasting tone."
            )
        else:
            collection_name = "realtime_products"
            system_instruction = (
                "You are a helpful retail store assistant. Mention prices clearly in INR, "
                "and summarize product specifications nicely for the customer."
            )

        # 🚀 BYPASS GOOGLE EMBEDDING BLOCKS: Create a deterministic vector locally
        words = query.lower().split()
        hash_vector = [0.0] * 1536
        for i, word in enumerate(words[:1536]):
            val = sum(ord(c) for c in word) / 1000.0
            hash_vector[i] = val
        
        query_vector = hash_vector

        # 2. Direct API call to Qdrant Cloud
        qdrant_host = os.getenv("QDRANT_HOST").rstrip("/")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        url = f"{qdrant_host}/collections/{collection_name}/points/search"
        headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
        payload = {
            "vector": query_vector,
            "limit": 2,
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

        # 3. Direct REST call to Gemini Text Generation
        gemini_key = os.getenv("GEMINI_API_KEY")
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        
        system_prompt = (
            f"{system_instruction}\n\n"
            f"Live Database Context Data:\n{context_text}\n\n"
            f"User Query: {query}\n"
            f"Answer:"
        )
        
        gen_payload = {
            "contents": [{"parts": [{"text": system_prompt}]}]
        }
        
        gen_res = requests.post(gen_url, json=gen_payload, headers={"Content-Type": "application/json"})
        
        if gen_res.status_code == 200:
            ai_answer = gen_res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            ai_answer = f"Gemini Text Generation Error: {gen_res.text}"
            
    except Exception as e:
        ai_answer = f"Search Loop Exception: {str(e)}"

    return {
        "user_query": query,
        "retrieved_context": context_chunks,
        "ai_response": ai_answer
    }

# 🌟 FIXED: Added the required main block back for Render to run on the correct port
if __name__ == "__main__":
    import uvicorn
    # Render reads the PORT environment variable automatically
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)