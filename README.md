# RAG-Based Patent Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying and exploring Sustainable Development Goal (SDG)-tagged patent data.  
Built in Python with:

- **FAISS** for vector retrieval over patent text chunks  
- **Sentence-Transformers** for embedding generation  
- **Mixtral API** (or similar) for LLM-based prompt rewriting, summarization, and final answer generation  
- Modular pipeline covering query rewriting, targeted retrieval, multi-stage summarization, and conversational context
- **NEW: Modern React Frontend** for an intuitive user experience

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

Here’s a complete `README.md` with **step-by-step copy-paste instructions** tailored to your RAG-based SDG patent chatbot project.

You can copy this and save it as `README.md` in the root of your GitHub repo:

---


## 🖥️ Setup Instructions (Windows-friendly)

> ✅ Follow these steps in order in **PowerShell** or **Command Prompt**

---

### 1. Clone the repository

```bash
git clone https://github.com/arnab013/RAGChabot.git
cd RAGChabot
````

---

### 2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # for Windows
```

---

### 3. Install required packages

```bash
pip install -r requirements.txt
```

If you get an error related to `parquet` or `pyarrow`, run:

```bash
pip install pyarrow
```

---

### 4. Prepare your dataset

Place your patent dataset in CSV format (e.g., `final_dataset.csv`) in the project root.

---

### 5. Build the embeddings & FAISS index

```bash
python -m src.embed_build final_dataset.csv
```

This will:

* Convert the CSV into a `.parquet` file for fast access
* Generate sentence embeddings for patent chunks
* Save a FAISS index and metadata in the `embeddings/` folder

---

### 6. Start the chatbot (Choose one option)

#### Option A: CLI Interface

```bash
python -m src.demo_cli
```

#### Option B: Web Interface with React Frontend

1. Start the Flask API server:
   ```bash
   python src/api.py
   ```

2. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

4. Start the React development server:
   ```bash
   npm start
   ```

5. Open your browser and go to http://localhost:3000

---

## React Frontend

The project now includes a modern, responsive React frontend built with Material UI. Features include:

- Clean, intuitive chat interface
- User and bot message styling
- Markdown support for formatted responses
- Loading indicators
- Clear chat functionality
- Mobile-friendly responsive design

### Build for Production

To build the React app for production:

```bash
cd frontend
npm run build
```

This creates an optimized build in the `frontend/build` directory that can be served by any web server.

---

## 💬 Example Queries

```text
👉 patents related to sdg 6 water purification in africa
👉 show me the top 10 technologies in sdg 3 with number of patents
👉 who is the applicant of patent (1487386)
👉 what is new in this patent according to the inventor
👉 how I can innovate from the Improved Medical Thermal Energy Exchange patent
👉 give me SDG 7 hydrogen-production patents filed after 2021
👉 list SDG 13 filings on direct-air CO₂ capture
```

---

## 📁 Project Structure

```
.
├── embeddings/            # FAISS index + metadata
├── src/
│   ├── demo_cli.py        # CLI entrypoint
│   ├── embed_build.py     # Builds embeddings and index
│   ├── retrieval.py       # FAISS chunk retriever
│   ├── pipeline.py        # RAG orchestration
│   ├── query_rewrite.py   # LLM-based rewrite + filter extraction
│   ├── summarise.py       # Map-reduce summarization
│   ├── stats_engine.py    # Yearly/group aggregation
│   ├── llm_clients.py     # Mixtral API handler
│   ├── filter_ops.py      # Applies dynamic filters
│   ├── token_utils.py     # Token counter
│   └── data_ingest.py     # Loads CSV/parquet and joins text
├── final_dataset.csv      # Your patent CSV (you provide this)
├── requirements.txt
└── README.md
```

---

## ✅ GitHub Deployment Steps

If you haven’t pushed yet:

```bash
git init
git add .
git commit -m "Initial Commit - a minimal RAG bot based on SDG Patent Data"
git remote add origin https://github.com/arnab013/RAGChabot.git
git branch -M main
git push -u origin main
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Developed by **Arnab Saha**
