import os
import requests
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Dynamically locate the .env file in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(title="Hybrid Multi-Domain RAG Search Engine")

# Local standard text embedder model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ai = genai.Client()

@app.get("/search")
async def hybrid_rag_search(
    query: str = Query(..., description="User query text"),
    domain: str = Query("products", description="Domain to search inside: 'products' or 'news'")
):
    print(f"🔍 Searching domain '{domain}' vector space for: '{query}'")
    
    context_chunks = []
    
    try:
        # Swap collection name and specialized conversational prompts based on domain choice
        if domain == "news":
            collection_name = "global_breaking_news"
            system_instruction = (
                "You are a live, elite breaking news bulletin assistant. Summarize current regional events "
                "based strictly on the live news flashes provided below. Speak in a neutral, clear, professional "
                "broadcasting tone. Do not copy-paste raw database strings; present it like a real news anchor."
            )
        else:
            collection_name = "realtime_products"
            system_instruction = (
                "You are a friendly, warm, and helpful real-time store assistant. Talk naturally, "
                "mention prices clearly in INR, and summarize product specifications for the customer. "
                "Do not copy-paste raw database values; behave like an expert human retail specialist."
            )

        # 1. Standard vector extraction via our safe local model layer
        query_vector = model.encode(query).tolist()

        # 2. Direct REST HTTP API call bypassing the broken qdrant-client wrapper completely!
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
        else:
            print(f"⚠️ Qdrant REST Error: {response.text}")
        
        context_text = "\n".join(context_chunks) if context_chunks else "No dynamic cloud updates matches this exact intent right now."
        print("📝 Context:\n", context_text)

        system_prompt = (
            f"{system_instruction}\n\n"
            f"Live Database Context Data:\n{context_text}\n\n"
            f"User Query: {query}\n"
            f"Answer:"
        )

        gemini_response = ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt,
        )
        ai_answer = gemini_response.text.strip()
            
    except Exception as e:
        ai_answer = f"Search Loop Exception: {str(e)}"

    return {
        "user_query": query,
        "retrieved_context": context_chunks,
        "ai_response": ai_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)