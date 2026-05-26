import streamlit as st
import requests
import os

st.set_page_config(page_title="Universal AI Live Assistant", page_icon="🤖", layout="centered")

st.title("🧠 Real-Time RAG Search Engine")
st.markdown("---")

# Dropdown select framework feature
search_domain = st.selectbox(
    "Choose search mode category:",
    ("E-Commerce Inventory Store", "Breaking News Radar Updates")
)

domain_key = "products" if search_domain == "E-Commerce Inventory Store" else "news"
placeholder_text = "e.g., I need a noise canceling headset..." if domain_key == "products" else "e.g., Any updates on infrastructure or traffic issues in Delhi?"

user_query = st.text_input(label="Ask your natural language question:", placeholder=placeholder_text)

# 🌟 DYNAMIC BACKEND OVERRIDE FOR CLOUD DEPLOYMENT
# It will dynamically fetch the correct independent backend server URL
BACKEND_BASE = os.getenv("BACKEND_URL", "https://real-time-rag-engine.onrender.com")
SEARCH_ENDPOINT = f"{BACKEND_BASE.rstrip('/')}/search"

if st.button("Query Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Type something first!")
    else:
        with st.spinner("Executing real-time semantic retrieval handshake..."):
            try:
                # Direct hit to the correct isolated backend URL
                response = requests.get(
                    SEARCH_ENDPOINT,
                    params={"query": user_query, "domain": domain_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("🎯 Semantic Synced Context Gathered!")
                    with st.expander("👀 View Fetched Live DB Context"):
                        context_list = data.get("retrieved_context", [])
                        if context_list:
                            for context in context_list:
                                st.info(context)
                        else:
                            st.write("No direct dynamic context matches found.")
                            
                    st.markdown("### 🤖 Engine Output Answer:")
                    st.write(data.get("ai_response"))
                else:
                    st.error(f"Backend API Error: Status {response.status_code} - Iska matlab frontend galat raste par jaa raha hai.")
            except Exception as e:
                st.error(f"Bridge connection broken: {str(e)}")