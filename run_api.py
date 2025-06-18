#!/usr/bin/env python3
"""
Run script for the API server with proper path handling
"""
import os
import sys

# Add the project root to the Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Also add src directory to Python path for direct imports
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, src_dir)

# Fix imports in API file
import src.api as api_module
from src.config import BACKEND_PORT

if __name__ == "__main__":
    port = int(os.environ.get('PORT', BACKEND_PORT))
    print(f"🚀 Starting API server on port {port}")
    api_module.app.run(host='0.0.0.0', port=port, debug=True)
