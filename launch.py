#!/usr/bin/env python3
"""
Production startup script for RAG Chatbot
This script starts the backend API server in production mode.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    print("🚀 Starting RAG Chatbot in Production Mode...")
    print(f"📁 Project Directory: {project_root}")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ Error: .env file not found. Please create it with the required API keys.")
        sys.exit(1)
    
    # Check if database and embeddings exist
    data_dir = project_root / "data"
    embeddings_dir = project_root / "embeddings"
    
    if not data_dir.exists():
        print("⚠️  Warning: data directory not found. Some features may not work.")
    
    if not embeddings_dir.exists():
        print("⚠️  Warning: embeddings directory not found. Please run embedding generation first.")
    
    # Start the API server
    try:
        print("🔥 Starting API server...")
        subprocess.run([sys.executable, "run_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
