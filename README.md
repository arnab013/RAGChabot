# GoalDigger - Patent Research Chatbot

GoalDigger is an intelligent chatbot designed for patent research and analysis, with a focus on Sustainable Development Goals (SDGs). The system uses Retrieval-Augmented Generation (RAG) technology to provide conversational access to patent databases, making complex patent research more accessible and intuitive.

## Features

### Conversation Management
GoalDigger maintains context throughout your conversation, remembering previous questions and building on past discussions. Whether you're diving deep into specific patents or asking casual questions, the bot adapts its responses accordingly.

### Database Analytics
The system provides comprehensive insights into patent databases, including breakdowns by country, year, technology domain, inventors, and companies. You can ask questions like "How many patents are from Germany?" or "Show me trends in renewable energy patents over the last decade."

### Advanced Patent Search
Search through patents using natural language queries. The system understands SDG classifications, patent codes (IPC/CPC), date ranges, and inventor names. Results are formatted with standardized European Patent (EP) numbering for easy reference.

### Modern Web Interface
Built with React and Tailwind CSS, the interface provides a clean, responsive chat experience. The frontend supports markdown formatting, making technical content easy to read on both desktop and mobile devices.

### Technical Implementation
- Uses Mixtral LLM for generating responses up to 5,000 tokens
- FAISS vector search for efficient similarity matching across patent embeddings
- Handles multiple query types: patent innovation analysis, claim summaries, and prior art research
- Session management keeps conversation context active across multiple interactions

## Getting Started

### What You'll Need
- Python 3.8 or newer
- Node.js 16 or newer
- Git for cloning the repository

### Installation

**1. Clone and Set Up the Project**

```bash
git clone https://github.com/arnab013/RAGChabot.git
cd RAGChabot

# Create a virtual environment
python -m venv ragbot
ragbot\Scripts\activate  # Windows
# source ragbot/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
```

**2. Configure API Keys**

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mixtral_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

You can get your Mistral API key from the [Mistral AI Platform](https://console.mistral.ai/). The SECRET_KEY can be any random string for session management.

**3. Prepare Your Data**

Place your patent dataset as `final_dataset.csv` in the project root, then build the search index:

```bash
python -m src.embed_build final_dataset.csv
```

**4. Start the Application**

Run the backend (Flask API):
```bash
cd src
python api.py
```
The backend will start on `http://localhost:5000`

In a new terminal, run the frontend (React):
```bash
cd frontend
npm install
npm start
```
The frontend will start on `http://localhost:3000`

Visit `http://localhost:3000` to start chatting with GoalDigger.

**Tip**: Keep both terminal windows open to monitor logs from both the backend and frontend.

## How to Use GoalDigger

GoalDigger is designed to understand natural language queries about patents. Here are some examples of what you can ask:

### Database Questions
```
"How many patents are in the database?"
"Show me patents by country distribution"
"What technologies are covered in the database?"
"Give me a breakdown of patents by year"
"Who are the top inventors in the database?"
"Which companies have the most patents?"
"Show me patent trends by decade"
"How many patents relate to each SDG?"
```

### Patent Research
```
"Find patents related to SDG 6 water purification in Africa"
"Show me SDG 7 renewable energy patents from 2020-2024"
"Patents by Tesla on battery technology and energy storage"
"What are the latest innovations in carbon capture and storage?"
"Patents about artificial intelligence in healthcare applications"
"Search for patents on sustainable agriculture techniques"
"Find solar panel efficiency patents from German inventors"
```

### Technical Analysis
```
"How can I innovate on patent EP1234567?"
"Summarize the claims of this patent"
"What's the prior art for patent EP7654321?"
"Who are the inventors of this technology?"
"Explain the SDG relevance of these patents"
```

### Casual Conversation
```
"Hi GoalDigger, how are you today?"
"What's your expertise area?"
"Thanks for the detailed analysis!"
"Can you help me understand patent landscapes?"
"What makes you different from other chatbots?"
"How do you stay so knowledgeable about patents?"
```

## How the System Works

The application is built with a React frontend that communicates with a Flask API backend. The backend processes queries through a RAG pipeline that combines patent database search with language model generation.

```
React Frontend ←→ Flask API ←→ RAG Pipeline
                        ↓
                  Conversation History ←→ Patent Database
                                              ↓
                                        FAISS Index
                                        Patent CSV
                                        Embeddings
```

### Project Structure

```
RAGChabot/
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/         # Chat interface components
│   │   ├── App.js             # Main application
│   │   └── index.css          # Tailwind styles
│   ├── package.json           # Node dependencies
│   └── tailwind.config.js     # Tailwind configuration
│
├── src/                       # Python backend
│   ├── api.py                 # Flask REST API
│   ├── pipeline.py            # RAG orchestration
│   ├── retrieval.py           # FAISS search engine
│   ├── llm_clients.py         # Mixtral integration
│   ├── query_rewrite.py       # Query preprocessing
│   ├── summarise.py           # Response generation
│   ├── embed_build.py         # Index building
│   ├── stats_engine.py        # Analytics engine
│   └── config.py              # Configuration
│
├── embeddings/                # Search index files
│   ├── faiss_chunks.idx       # FAISS vector index
│   ├── patents.parquet        # Processed patent data
│   └── meta.pkl               # Metadata cache
│
├── final_dataset.csv          # Raw patent dataset
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                  # Documentation
```

---

## ⚡ Performance & Optimization

### Database Size Recommendations
- **Small datasets** (< 1,000 patents): Instant responses
- **Medium datasets** (1,000 - 10,000 patents): < 2 second responses
- **Large datasets** (> 10,000 patents): 2-5 second responses
- **Very large datasets** (> 100,000 patents): Consider chunking or distributed setup

### Response Time Optimization
- **Vector search**: Usually < 100ms with FAISS
- **LLM generation**: 1-4 seconds depending on response length
- **Statistics queries**: < 500ms for most aggregations
- **Context retrieval**: Minimal overhead with session management

### Memory Usage
- **Base application**: ~200MB RAM
- **FAISS index**: ~10MB per 1,000 patents
- **LLM inference**: Handled by Mistral API (no local memory impact)
- **Session storage**: ~1KB per active session

## Development

### Adding New Features

To modify the system:
1. Backend changes go in the `src/` directory
2. Frontend updates are made in `frontend/src/components/`
3. Database query extensions should be added to `get_database_stats()` in `pipeline.py`
4. New LLM capabilities can be implemented by updating prompts in `llm_clients.py`

### Testing

```bash
# Test backend components
python -c "from src.pipeline import RAGPipeline; print('Backend OK')"

# Test API endpoints
python -c "from src.api import app; print('API OK')"

# Test frontend build
cd frontend && npm run build

# Test database statistics
python -c "from src.pipeline import RAGPipeline; r = RAGPipeline('final_dataset.csv'); print(r.get_database_stats())"
```

### Common Issues

**Import errors**: Make sure you're in the correct directory and your virtual environment is activated.

**Missing API key**: Check that your `.env` file exists and contains a valid `MISTRAL_API_KEY`.

**Port conflicts**: If ports 3000 or 5000 are in use, you can change them in `package.json` (React) or `api.py` (Flask).

**FAISS index missing**: Run `python -m src.embed_build final_dataset.csv` to rebuild the search index.

**Frontend build issues**: Try `npm cache clean --force` followed by `npm install`.

### Production Deployment

1. Build the frontend: `cd frontend && npm run build`
2. Set production environment variables in your `.env` file
3. Deploy using your preferred hosting service (Heroku, AWS, etc.)

## Advanced Usage

### Custom Datasets

GoalDigger works with patent datasets containing these columns:
- `title_en` or `title`: Patent titles
- `abstract_text`: Patent abstracts  
- `publication_date`: Publication dates (YYYY-MM-DD format)
- `applicant_names`: Company/organization names
- `inventor_names`: Inventor information
- `sdg_number`: SDG classifications (optional)
- `publication_number`: Patent numbers (optional, for EP formatting)

**Data Requirements:**
- CSV format with headers
- UTF-8 encoding recommended
- At least 100 patents for meaningful statistics
- Date format: YYYY-MM-DD or YYYY

### API Integration

The Flask API provides REST endpoints for integration:

**Chat Endpoint:**
```bash
POST /api/chat
Content-Type: application/json

{
    "query": "How many patents are in the database?",
    "session_id": "user-session-123"
}
```

**Response Format:**
```json
{
    "response": "Database Overview\n\nI currently have access to 15,847 patents in my database...",
    "session_id": "user-session-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Health Check:**
```bash
GET /api/health
```

**Session Management:**
- Sessions automatically expire after 1 hour of inactivity
- Context is maintained for up to 10 previous messages per session
- Use consistent `session_id` for conversation continuity

## Contributing

We welcome contributions to improve GoalDigger. Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test them
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Developer**: Arnab Saha
- **Issues**: [GitHub Issues](https://github.com/arnab013/RAGChabot/issues)
- **Documentation**: See inline code comments and this README

## Acknowledgments

Thanks to the following projects and communities that made GoalDigger possible:

- Mixtral AI for powerful language model capabilities
- FAISS for efficient vector search and similarity matching
- React and Tailwind CSS for the modern, responsive user interface
- Flask for lightweight and flexible API development
- The open source community for supporting tools and libraries

---

*Version 2.0 - Built for advancing patent research and sustainable development*
