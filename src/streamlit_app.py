import streamlit as st
import time
from retrieval import PassageRetriever
from pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="RAGBot - Smart Patent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7f9;
    }
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.8rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #2196F3;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        border-left: 5px solid #9c27b0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
    }
    .main-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        color: #333;
    }
    .subtitle {
        color: #666;
        font-style: italic;
        margin-top: 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103652.png", width=100)
with col2:
    st.markdown("<div class='main-header'><h1 class='main-title'>RAGBot</h1> <h2>🔎</h2></div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your smart patent research assistant powered by RAG technology</p>", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    debug_mode = st.checkbox("Debug Mode", value=True, help="Show additional information about the retrieval process")
    
    st.subheader("About")
    st.markdown("""
    **RAGBot** is a Retrieval Augmented Generation system for patent research.
    
    It helps you find relevant patent information by:
    1. Retrieving relevant passages from the patent database
    2. Analyzing and processing the information
    3. Generating concise, informative responses
    """)
    
    st.divider()
    st.caption("© 2025 CodeFest Summer Project")

# Initialize retriever and pipeline (cache to avoid reloads)
@st.cache_resource
def get_pipeline():
    retriever = PassageRetriever()
    pipeline = RAGPipeline(retriever, debug=debug_mode)
    return pipeline

pipeline = get_pipeline()

# Main container for chat interface
chat_container = st.container()

# Chat interface
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    
    # Add an intro message
    welcome_message = """
    👋 Hello! I'm RAGBot, your patent research assistant.
    
    You can ask me questions about patents, technologies, and innovations. For example:
    - "Tell me about recent advancements in quantum computing"
    - "What patents exist for solar panel efficiency improvements?"
    - "Summarize patents related to EV battery technology from 2020-2025"
    
    How can I assist you today?
    """
    st.session_state['chat_history'].append((None, welcome_message))

# User input
with st.form(key='chat_form', clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        user_input = st.text_input("Ask a question about patents:", placeholder="E.g., What are recent innovations in battery technology?", label_visibility="collapsed")
    with cols[1]:
        submit = st.form_submit_button("Send 🚀")

# Process the input
if submit and user_input.strip():
    # Add user message to chat
    st.session_state['chat_history'].append((user_input, None))
    
    # Create a placeholder for the bot's response and the typing indicator
    with chat_container:
        message_placeholder = st.empty()
        message_placeholder.markdown("*RAGBot is thinking...*")
        
        # Get response from the pipeline
        response = pipeline.ask(user_input)
        
        # Simulate typing effect
        message_placeholder.empty()
        
        # Add bot response to chat history
        st.session_state['chat_history'][-1] = (user_input, response)
        
        # Force a rerun to update the display immediately
        st.experimental_rerun()

# Display chat history
with chat_container:
    for i, (user, bot) in enumerate(st.session_state['chat_history']):
        if user:
            # User message
            st.markdown(f"""
            <div class="chat-message user">
                <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png" class="avatar">
                <div class="message">
                    <b>You</b><br>{user}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if bot:
            # Bot message
            st.markdown(f"""
            <div class="chat-message bot">
                <img src="https://cdn-icons-png.flaticon.com/512/2103/2103652.png" class="avatar">
                <div class="message">
                    <b>RAGBot</b><br>{bot}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Clear chat button
if st.session_state['chat_history']:
    if st.button("Clear Chat"):
        st.session_state['chat_history'] = []
        st.experimental_rerun()
