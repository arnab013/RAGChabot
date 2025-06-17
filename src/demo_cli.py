#!/usr/bin/env python
import sys
from retrieval import PassageRetriever
from pipeline  import RAGPipeline

def main():
    # no CSV argument needed
    retriever = PassageRetriever()             # will auto-load parquet + index
    pipeline  = RAGPipeline(retriever, debug=True)

    print("🔎 RAG chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("👉  ").strip()
        if not q or q.lower() in ("exit","quit"):
            break
        print(pipeline.ask(q))

if __name__ == "__main__":
    main()
