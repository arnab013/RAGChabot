# RAG-Based Patent Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying and exploring Sustainable Development Goal (SDG)-tagged patent data.  
Built in Python with:

- **FAISS** for vector retrieval over patent text chunks  
- **Sentence-Transformers** for embedding generation  
- **Mixtral API** (or similar) for LLM-based prompt rewriting, summarization, and final answer generation  
- Modular pipeline covering query rewriting, targeted retrieval, multi-stage summarization, and conversational context

---

## Features

- **Conversational context**: retains the last 5 turns  
- **Intelligent query pre-processing**: rewrites user questions for clarity and extracts structured filters & column priorities  
- **Structured filters**: SDG numbers, IPC/CPC codes, publication dates, applicant/inventor countries, etc.  
- **Chunk-level indexing**: breaks long documents into 512-token chunks with overlap for fine-grained recall  
- **Map-reduce summarization**: handles large contexts by summarizing in stages  
- **Special-case branches**: inventor insights, claim summarization, prior art lookup, family lookup, aggregation & time-series  
- **“Innovate on patent”** and **“Brainstorm on topic”** modes for R&D prompts  

---

## Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/arnab013/RAGChabot.git
   cd RAGChabot
