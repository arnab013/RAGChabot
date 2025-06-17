# 🤖 GoalDigger - Advanced RAG Patent Chatbot

**GoalDigger** is a sophisticated AI-powered chatbot specializing in patent research and Sustainable Development Goals (SDGs). Built with cutting-edge Retrieval-Augmented Generation (RAG) technology, GoalDigger provides intelligent, conversational access to patent databases with comprehensive analysis capabilities.

## ✨ Key Features

### 🧠 **Intelligent Conversation Management**
- **Session-based context**: Maintains conversation history across patent queries and casual chat
- **Dynamic personality**: Sassy, confident, and helpful GoalDigger persona with professional expertise
- **Context-aware responses**: Remembers previous discussions and references them appropriately
- **Seamless mode switching**: Transitions between patent research and casual conversation naturally

### 📊 **Advanced Database Analytics**
- **Comprehensive statistics**: Get insights on patent counts, distributions, and trends
- **Multi-dimensional analysis**: Breakdown by country, year, technology, inventor, and company
- **Smart categorization**: Automatic technology domain classification from patent content
- **Temporal analysis**: Decade-wise trends and publication timeline insights

### 🔍 **Sophisticated Patent Search**
- **Natural language queries**: Ask questions in plain English about patents and SDGs
- **Intelligent filtering**: SDG numbers, IPC/CPC codes, dates, countries, inventors
- **Chunk-level precision**: 512-token chunks with overlap for fine-grained retrieval
- **Multi-stage summarization**: Map-reduce approach for handling large contexts

### 🎨 **Modern User Experience**
- **React frontend**: Beautiful, responsive web interface with Tailwind CSS
- **Real-time chat**: Instant responses with typing indicators and message history
- **Markdown support**: Rich formatting for technical content and patent details
- **Mobile-friendly**: Optimized for both desktop and mobile devices

### 🛠 **Advanced Technical Capabilities**
- **Mixtral LLM integration**: State-of-the-art language model for accurate responses
- **FAISS vector search**: High-performance similarity search over patent embeddings
- **Special query modes**: Patent innovation, claim analysis, prior art lookup
- **5000-token responses**: Comprehensive, detailed answers without cutoffs

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone & Setup Environment

```bash
git clone https://github.com/arnab013/RAGChabot.git
cd RAGChabot

# Create virtual environment
python -m venv ragbot
ragbot\Scripts\activate  # Windows
# source ragbot/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mixtral_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

### 3. Prepare Patent Data

Place your patent dataset as `final_dataset.csv` in the project root, then build the search index:

```bash
python -m src.embed_build final_dataset.csv
```

### 4. Start the Application

**Backend (Flask API):**
```bash
cd src
python api.py
```

**Frontend (React UI):**
```bash
cd frontend
npm install
npm start
```

Access the application at `http://localhost:3000`

---

## 💬 Query Examples

### 📊 **Database Statistics**
```
🔸 "How many patents are in the database?"
🔸 "Show me patents by country"
🔸 "What technologies are covered?"
🔸 "Patents by year breakdown"
🔸 "Top inventors in the database"
🔸 "Which companies have the most patents?"
```

### 🔍 **Patent Research**
```
🔸 "Patents related to SDG 6 water purification in Africa"
🔸 "Show me SDG 7 renewable energy patents from 2020-2024"
🔸 "Find patents by Tesla on battery technology"
🔸 "What are the latest innovations in carbon capture?"
🔸 "Patents about AI in healthcare applications"
```

### 🧪 **Innovation & Analysis**
```
🔸 "How can I innovate on patent EP1234567?"
🔸 "Summarize the claims of this patent"
🔸 "What's the prior art for patent EP7654321?"
🔸 "Who are the inventors of this technology?"
🔸 "Explain the SDG relevance of these patents"
```

### 💭 **Casual Conversation**
```
🔸 "Hi GoalDigger, how are you today?"
🔸 "What's your expertise area?"
🔸 "Thanks for the detailed analysis!"
🔸 "Can you help me understand patent landscapes?"
```

---

## 🏗 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React UI      │◄──►│   Flask API      │◄──►│   RAG Pipeline  │
│                 │    │                  │    │                 │
│ • Chat Interface│    │ • Session Mgmt   │    │ • Query Analysis│
│ • Markdown      │    │ • Context Track  │    │ • FAISS Search  │
│ • Responsive    │    │ • Error Handling │    │ • LLM Generation│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Conversation   │    │   Data Layer    │
                       │    History      │    │                 │
                       │                 │    │ • FAISS Index   │
                       │ • Session Store │    │ • Patent CSV    │
                       │ • Context Mgmt  │    │ • Embeddings    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
RAGChabot/
├── 📁 frontend/                 # React application
│   ├── 📁 src/
│   │   ├── 📁 components/       # Chat UI components
│   │   ├── App.js              # Main app component
│   │   └── index.css           # Tailwind styles
│   ├── package.json            # Node dependencies
│   └── tailwind.config.js      # Tailwind configuration
│
├── 📁 src/                     # Python backend
│   ├── api.py                  # Flask REST API
│   ├── pipeline.py             # RAG orchestration
│   ├── retrieval.py            # FAISS search engine
│   ├── llm_clients.py          # Mixtral integration
│   ├── query_rewrite.py        # Query preprocessing
│   ├── summarise.py            # Response generation
│   ├── embed_build.py          # Index building
│   ├── stats_engine.py         # Analytics engine
│   └── config.py               # Configuration
│
├── 📁 embeddings/              # Search index files
│   ├── faiss_chunks.idx        # FAISS vector index
│   ├── patents.parquet         # Processed patent data
│   └── meta.pkl                # Metadata cache
│
├── final_dataset.csv           # Raw patent dataset
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

---

## 🔧 Development

### Adding New Features

1. **Backend changes**: Modify files in `src/`
2. **Frontend updates**: Edit React components in `frontend/src/`
3. **Database queries**: Extend `get_database_stats()` in `pipeline.py`
4. **New LLM capabilities**: Update prompts in `llm_clients.py`

### Testing

```bash
# Test backend components
python -c "from src.pipeline import RAGPipeline; print('Backend OK')"

# Test API endpoints
python -c "from src.api import app; print('API OK')"

# Test frontend build
cd frontend && npm run build
```

### Production Deployment

1. **Build frontend**: `cd frontend && npm run build`
2. **Configure environment**: Set production API keys
3. **Deploy**: Use your preferred hosting service (Heroku, AWS, etc.)

---

## 🎯 Advanced Usage

### Custom Datasets

GoalDigger supports any patent dataset with these columns:
- `title_en` or `title`: Patent titles
- `abstract_text`: Patent abstracts
- `publication_date`: Publication dates
- `applicant_names`: Company/organization names
- `inventor_names`: Inventor information
- `sdg_number`: SDG classifications (optional)

### API Integration

The Flask API provides REST endpoints:

```bash
POST /api/chat
{
    "query": "How many patents are in the database?",
    "session_id": "user-session-123"
}
```

Response:
```json
{
    "response": "📊 Database Overview\n\nI currently have access to **15,847 patents** in my database...",
    "session_id": "user-session-123"
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Support & Contact

- **Developer**: Arnab Saha
- **Issues**: [GitHub Issues](https://github.com/arnab013/RAGChabot/issues)
- **Documentation**: See inline code comments and this README

---

## 🙏 Acknowledgments

- **Mixtral AI** for language model capabilities
- **FAISS** for efficient vector search
- **React & Tailwind** for the modern UI
- **Open source community** for supporting tools and libraries

---


