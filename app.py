import streamlit as st
import requests

st.set_page_config(page_title="Universal AI Live Assistant", page_icon="🤖", layout="centered")

st.title("🧠 Real-Time AI Search Portal")
st.markdown("---")

# Dropdown select framework feature
search_domain = st.selectbox(
    "Choose search mode category:",
    ("E-Commerce Inventory Store", "Breaking News Radar Updates")
)

domain_key = "products" if search_domain == "E-Commerce Inventory Store" else "news"
placeholder_text = "e.g., I need a noise canceling headset..." if domain_key == "products" else "e.g., Any updates on infrastructure or traffic issues in Delhi?"

user_query = st.text_input(label="Ask your natural language question:", placeholder=placeholder_text)

if st.button("Query Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Type something first!")
    else:
        with st.spinner("Executing real-time semantic retrieval handshake..."):
            try:
                response = requests.get(
                    f"http://localhost:8001/search",
                    params={"query": user_query, "domain": domain_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("🎯 Semantic Synced Context Gathered!")
                    with st.expander("👀 View Fetched Live DB Context"):
                        for context in data.get("retrieved_context", []):
                            st.info(context)
                            
                    st.markdown("### 🤖 Engine Output Answer:")
                    st.write(data.get("ai_response"))
                else:
                    st.error(f"Backend API Error: Status {response.status_code}")
            except Exception as e:
                st.error(f"Bridge connection broken: {str(e)}")