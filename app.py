import streamlit as st
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Pharmacogenomics RAG", layout="wide")

st.title("Pharmacogenomics RAG System")
st.markdown("""
Ask questions about:
- Gene function and variants  
- Drug metabolism  
- Gene–drug-variant interactions  
Uses PubMed Central research articles for grounded answers.<br><br>
⚠️ Demo version: This app uses ~490 PMC papers and focuses on pharmacogenomics of Warfarin and Isoniazid. Results are limited to this dataset.

**Features:** PubMed Central knowledge base | Biomedical embeddings (PubMedBERT) | FAISS semantic search | Groq LLM inference
""", unsafe_allow_html=True)

# -----------------------------
#  SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")

user_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen/qwen3-32b"]
)

gene_filter = st.sidebar.text_input("Filter by Gene (optional)")
drug_filter = st.sidebar.text_input("Filter by Drug (optional)")

st.sidebar.markdown("---")
st.sidebar.info("Built with PubMed Central + RAG pipeline")

# -----------------------------
#  LOAD DATA
# -----------------------------
@st.cache_resource
def load_all():
    index = faiss.read_index("faiss_index.bin")

    with open("metadata.json") as f:
        chunks = json.load(f)

    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    return index, chunks, model

index, chunks, model = load_all()

# -----------------------------
#  SEARCH FUNCTION
# -----------------------------
def search(query, top_k=5, gene_filter=None, drug_filter=None, score_threshold=0.3):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k * 3)

    results = []

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        score = float(distances[0][i])
        if score < score_threshold:
            continue

        chunk = chunks[idx]

        if gene_filter and gene_filter not in chunk.get("genes", []):
            continue

        if drug_filter and drug_filter not in chunk.get("drugs", []):
            continue

        results.append({
            "pmcid": chunk["pmcid"],
            "section": chunk["section"],
            "score": score,
            "text": chunk["text"][:300],
            "genes": chunk.get("genes", []),
            "drugs": chunk.get("drugs", [])
        })

        if len(results) >= top_k:
            break

    return results

# -----------------------------
# BUILD CONTEXT
# -----------------------------
def build_context(results):
    context = ""
    for i, r in enumerate(results):
        context += f"""
[Source {i+1}]
PMCID: {r['pmcid']}
Section: {r['section']}
Text: {r['text']}
"""
    return context

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns([3,1])

with col1:
    st.markdown("""
    <div style='margin-bottom:10px; font-size:18px; font-weight:600;'>
    🔍 Enter your question
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="e.g. How does CYP2C9 affect warfarin metabolism?"
    )

with col2:
    st.markdown("<div style='height:35px;'></div>", unsafe_allow_html=True)
    top_k = st.slider("Results", 1, 10, 3)
# -----------------------------
# RUN PIPELINE
# -----------------------------
if query:
    if not user_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar")
        st.stop()

    client = Groq(api_key=user_api_key)

    def ask_groq(prompt):
        response = client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content

    with st.spinner("🔬 Searching and generating answer..."):
        results = search(
            query,
            top_k=top_k,
            gene_filter=gene_filter,
            drug_filter=drug_filter
        )

        if not results:
            st.warning("⚠️ No relevant results found. Try a different query.")
            st.stop()

        context = build_context(results)

        # Query-type awareness
        if "what is" in query.lower():
            mode_instruction = "Start with a clear definition."
        else:
            mode_instruction = "Focus on mechanism and biological relationships."

        prompt = f"""
You are a pharmacogenomics expert.

{mode_instruction}

Context:
{context}

Question:
{query}

Instructions:
- Provide a clear scientific explanation
- Mention relevant genes and drugs explicitly
- Avoid irrelevant experimental details
- Cite sources inline using [PMCID]
- Keep answer concise (5–6 sentences max)
- If missing info, say "insufficient information"

Answer:
"""

        answer = ask_groq(prompt)

    # -----------------------------
    # ANSWER BOX
    # -----------------------------
    st.subheader("Answer")

    st.markdown(f"""
    <div style='
        padding:15px;
        border-radius:10px;
        background-color:#323;
        color:white;
        font-size:16px;
        line-height:1.6;
    '>
    {answer}
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # SOURCES 
    # -----------------------------
    st.subheader("Sources")

    unique_sources = sorted(set(r["pmcid"] for r in results))

    cols = st.columns(len(unique_sources))
    for i, pmcid in enumerate(unique_sources):
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        cols[i].markdown(f"[{pmcid}]({url})")

    # -----------------------------
    # CONTEXT VIEWER
    # -----------------------------
    with st.expander("🔬 Retrieved Context"):
        for r in results:
            st.write(f"**{r['pmcid']} ({r['section']})**")
            st.write(f"Score: {r['score']:.3f}")
            st.write(r["text"])
            st.write("---")