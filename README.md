# Pharmacogenomics RAG System

A biomedical **Retrieval-Augmented Generation (RAG)** application that answers pharmacogenomics questions using **PubMed Central (PMC)** research articles.

---

## Features

- 📚 PubMed Central knowledge base (~490 papers)
- 🧬 Biomedical embeddings (PubMedBERT)
- 🔍 Semantic search using FAISS
- 🤖 LLM inference via Groq API
- 🎯 Gene and drug-based filtering
- 🔗 Clickable research sources

---

## Demo Scope

⚠️ This is a **demo application**:

- Focus drugs: **Warfarin** and **Isoniazid**
- Dataset: ~490 PMC articles
- Results are limited to indexed data

---

## Example Queries

- What is NAT2?
- How does CYP2C9 affect warfarin metabolism?
- Which genes influence isoniazid toxicity?
- which varint associated with isoniazid toxicity ?
- How do NAT2 polymorphisms affect isoniazid toxicity?
- What is the role of VKORC1 in warfarin response?
- Which genes influence warfarin dose variability?

---

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers (PubMedBERT)
- Groq API

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py