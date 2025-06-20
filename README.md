# Patent Research and Analysis Platform

A comprehensive system for intelligent patent search and analysis, leveraging advanced natural language processing and retrieval-augmented generation (RAG) technologies to provide accurate, contextual information about patent data.

## Overview

This platform combines modern AI capabilities with structured patent databases to deliver a professional-grade research tool. The system automatically classifies user queries, performs semantic searches across patent documents, and generates comprehensive responses with supporting data visualizations.

## Key Features

- **Advanced Query Processing**: Intelligent classification of natural language queries into appropriate response categories
- **Semantic Patent Search**: Context-aware search capabilities across comprehensive patent databases
- **Interactive Analytics**: Statistical analysis tools with dynamic data visualization
- **Professional Web Interface**: Modern, responsive frontend built with React
- **RESTful API**: Well-documented API endpoints for integration and extensibility
- **Production Ready**: Scalable architecture with comprehensive error handling and logging

## Technical Architecture

### Core Components

**API Server** (`src/api.py`)
- Flask-based REST API with session management
- Request routing and response formatting
- Authentication and security middleware

**Query Classification Engine** (`src/query_classifier.py`)
- Natural language understanding for query categorization
- Machine learning-based intent detection
- Context-aware query processing

**Semantic Search Module** (`src/sql_retriever.py`)
- Vector-based document retrieval
- Remote embedding API integration
- Optimized search algorithms

**Language Model Integration** (`src/llm_clients.py`)
- Google Gemini API integration
- Response generation and formatting
- Model switching capabilities

**Analytics Engine** (`src/stats_queries.py`)
- Statistical analysis of patent data
- Data aggregation and visualization
- Trend analysis capabilities

### Frontend Application

**React Interface** (`frontend/`)
- Component-based architecture
- Real-time chat interface
- Dynamic content rendering
- Responsive design implementation

## Installation and Setup

### System Requirements

- Python 3.8 or higher
- Node.js 14.x or higher
- Google Gemini API access
- Remote embedding service access

### Installation Steps

1. **Repository Setup**
   ```bash
   git clone <repository-url>
   cd RAGChabot
   ```

2. **Backend Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_MODEL=gemini-2.0-flash
   REMOTE_EMBEDDING_URL=https://api.confusedelectrons.xyz/embed-query-w-sentence-transformers/
   REMOTE_EMBEDDING_API_KEY=your_embedding_api_key
   BACKEND_PORT=5000
   FRONTEND_PORT=3000
   ```

4. **Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

5. **Sample Data (Optional)**
   
   To generate fresh sample data for testing:
   ```bash
   python generate_sample_data.py
   ```
   
   This creates `data/sample_patent_data.xlsx` with 100 fictional patent records for demonstration purposes.

### Running the Application

1. **Start Backend Server**
   ```bash
   python launch.py
   ```
   
   Alternative method:
   ```bash
   python run_api.py
   ```
   
   The server will be available at `http://localhost:5000`

2. **Start Frontend Development Server**
   ```bash
   cd frontend
   npm start
   ```
   
   The application will open at `http://localhost:3000`

## Language Model Configuration

The system utilizes Google Gemini models for natural language processing and response generation. Model selection can be configured through environment variables.

### Supported Models

- `gemini-2.0-flash-exp` - Latest experimental model with enhanced capabilities
- `gemini-2.0-flash` - Stable production model (recommended)
- `gemini-1.5-pro` - High-performance model for complex queries
- `gemini-1.5-flash` - Optimized for speed and efficiency
- `gemini-1.0-pro` - Proven stable model

### Model Configuration

Update the model selection in your `.env` file:
```bash
GOOGLE_MODEL=gemini-1.5-pro
```

Restart the application to apply configuration changes.

## API Reference

### Search Endpoint

**POST** `/api/search`

Submit queries for patent search and analysis.

**Request Format:**
```json
{
  "query": "Your search query or question"
}
```

**Response Format:**
```json
{
  "message": "Response content",
  "chart": null,
  "error": "",
  "insight": "",
  "takeaway": ""
}
```

### Conversation Management

**POST** `/api/reset`

Reset the current conversation session.

**Response:**
```json
{
  "message": "Conversation reset successfully"
}
```

## System Configuration

### Database Setup

The system requires a SQLite database containing pre-processed patent data with embedded vectors. A sample dataset is provided at `data/sample_patent_data.xlsx` for demonstration purposes.

**Sample Data**: The repository includes a sample dataset with 100 fictional patent records to demonstrate the system functionality and data structure. This allows you to:
- Test the application immediately
- Understand the expected data format
- Develop and validate new features

**Production Data**: For production use, replace the sample data with your actual patent dataset following the same structure and format.

The database should include:
- Patent metadata tables
- Pre-computed embedding vectors
- Indexed search capabilities

### Embedding Service

The system integrates with remote embedding services using sentence-transformer models. This ensures consistency with pre-embedded patent data and provides optimal search performance.

## Production Deployment

### Backend Deployment

1. **WSGI Server Configuration**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
   ```

2. **Infrastructure Requirements**
   - Reverse proxy configuration (nginx recommended)
   - SSL/TLS certificate setup
   - Environment variable management
   - Logging and monitoring systems

### Frontend Deployment

1. **Production Build**
   ```bash
   cd frontend
   npm run build
   ```

2. **Static Asset Serving**
   - Configure web server (nginx/Apache)
   - Set up CDN for static assets
   - Configure API proxy settings

### Security Considerations

- **HTTPS Implementation**: Mandatory for production environments
- **Rate Limiting**: API endpoint protection against abuse
- **Environment Security**: Secure management of API keys and sensitive data
- **CORS Configuration**: Appropriate cross-origin resource sharing settings
- **Regular Updates**: Maintain security patches and dependency updates

## Development Guide

### Project Structure

```
RAGChabot/
├── src/                     # Backend application code
│   ├── api.py              # Main Flask application
│   ├── query_classifier.py # Query processing logic
│   ├── sql_retriever.py    # Database search functionality
│   ├── llm_clients.py      # Language model integration
│   ├── config.py           # Configuration management
│   └── stats_queries.py    # Analytics and statistics
├── frontend/               # React frontend application
│   ├── src/               # React components and logic
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
├── data/                  # Database files and datasets
├── embeddings/            # Vector embeddings and indices
├── database/              # Database models and configuration
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

### Extension and Customization

**Adding Query Types**
Extend the `query_classifier.py` module to support additional query classifications and response patterns.

**Response Formatting**
Modify response handlers in `api.py` to support new data formats and visualization types.

**Frontend Components**
Develop new React components in the `frontend/src` directory for enhanced user interface features.

**Analytics Capabilities**
Extend the `stats_queries.py` module to include additional statistical analysis and reporting features.

## Technical Support

For technical issues, feature requests, or development questions, please refer to the project documentation or contact the development team.

## License

This project is licensed under the terms specified in the LICENSE file. Please review the license agreement before using or modifying the software.
