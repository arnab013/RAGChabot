# RAG Chatbot - Patent Search and Analysis System

A production-ready RAG (Retrieval-Augmented Generation) chatbot system for patent search and analysis, featuring intelligent query classification, semantic search, and conversational AI capabilities.

## Features

- **Intelligent Query Classification**: Automatically determines whether queries are conversational, patent search, or statistical requests
- **Patent Search**: Semantic search through patent database with formatted results
- **Conversational AI**: General chat capabilities with LLM integration
- **Statistical Analysis**: Patent data analysis and visualization (extensible)
- **Modern Web Interface**: React-based frontend with responsive design
- **Structured API Responses**: JSON-formatted responses with multiple content types

## System Architecture

### Backend Components
- **API Server** (`src/api.py`): Flask-based REST API with session management
- **Query Classifier** (`src/query_classifier.py`): LLM-powered query type detection
- **SQL Retriever** (`src/sql_retriever.py`): Database search with remote embedding API
- **LLM Clients** (`src/llm_clients.py`): Google Gemini Flash 2.0 integration for response generation
- **Statistics Engine** (`src/stats_queries.py`): Patent data analysis capabilities

### Frontend Components
- **React Application** (`frontend/`): Modern web interface
- **Chat Interface**: Real-time conversation with the AI
- **Response Rendering**: Support for text, semantic, and chart response types

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Google Gemini API key
- Remote embedding API access

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd RAGChabot
   ```

2. **Backend Setup**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**   Create `.env` file:   ```
   GOOGLE_API_KEY=your_google_api_key
   
   # Model Configuration (Google only)
   GOOGLE_MODEL=gemini-2.0-flash
   
   REMOTE_EMBEDDING_URL=https://api.confusedelectrons.xyz/embed-query-w-sentence-transformers/
   REMOTE_EMBEDDING_API_KEY=your_embedding_api_key
   BACKEND_PORT=5000
   FRONTEND_PORT=3000
   ```

4. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. **Start Backend Server (Production)**
   ```bash
   python launch.py
   ```
   Or manually:
   ```bash
   python run_api.py
   ```
   Server will start on http://localhost:5000

2. **Start Frontend Development Server**
   ```bash
   cd frontend
   npm start
   ```
   Application will open at http://localhost:3000

## Model Configuration

The system uses Google Gemini models exclusively. You can easily switch between different Gemini models through environment variables:

### Available Google Models

- `gemini-2.0-flash-exp` - Latest experimental Flash model  
- `gemini-2.0-flash` - Stable Flash model (Default)
- `gemini-1.5-pro` - Production-ready pro model
- `gemini-1.5-flash` - Fast, efficient model
- `gemini-1.0-pro` - Stable pro model

### Switching Models

Simply update your `.env` file:
```bash
# Switch to a different Google model
GOOGLE_MODEL=gemini-1.5-pro
```

Restart the application to apply changes.

## API Endpoints

### Chat Endpoint
**POST** `/api/chat`

**Request:**
```json
{
  "message": "Your query here"
}
```

**Response:**
```json
{
  "message": {
    "type": "text|semantic_text|chart_text",
    "content": {
      "title": "Response Title",
      "body": "Response content...",
      "matched_chunks": [] // For semantic_text type
    }
  },
  "session_id": "uuid",
  "query_types": ["conversation"],
  "classification": {
    "detected_types": ["CONVERSATION"],
    "confidence": 0.99,
    "reasoning": "Classification explanation"
  }
}
```

### Other Endpoints
- **GET** `/api/history` - Get conversation history
- **GET** `/api/session-info` - Get session information

## Configuration

### Database
The system uses SQLite with pre-embedded patent chunks. The database should be located at `data/patents.db`.

### Embedding Model
Uses remote embedding API with sentence-transformers model for consistency with pre-embedded data.

### LLM Configuration
Configured to use Mistral AI's open-mixtral-8x22b model for response generation.

## Production Deployment

### Backend Deployment
1. Use a production WSGI server (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
   ```

2. Set up reverse proxy (nginx recommended)
3. Configure environment variables for production
4. Enable logging and monitoring

### Frontend Deployment
1. Build production bundle:
   ```bash
   cd frontend
   npm run build
   ```

2. Serve static files with nginx or CDN
3. Configure API proxy settings for production backend

### Security Considerations
- Use HTTPS in production
- Implement rate limiting
- Secure API keys and environment variables
- Configure CORS appropriately
- Regular security updates

## Development

### Project Structure
```
RAGChabot/
├── src/                    # Backend source code
│   ├── api.py             # Main Flask application
│   ├── query_classifier.py # Query classification logic
│   ├── sql_retriever.py   # Database search functionality
│   ├── llm_clients.py     # LLM integration
│   ├── config.py          # Configuration settings
│   └── ...
├── frontend/              # React frontend application
│   ├── src/
│   ├── public/
│   └── package.json
├── data/                  # Database and data files
├── embeddings/           # Embedding models and indices
├── requirements.txt      # Python dependencies
└── README.md
```

### Adding New Features
1. **Query Types**: Extend `query_classifier.py` for new query classifications
2. **Response Types**: Add new response formatters in `api.py`
3. **Frontend Components**: Create new React components for response rendering

## License

[Add your license information here]

## Support

[Add support/contact information here]
