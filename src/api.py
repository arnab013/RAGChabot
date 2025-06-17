from flask import Flask, request, jsonify, session
from flask_cors import CORS
from retrieval import PassageRetriever
from pipeline import RAGPipeline
from llm_clients import chat
import os
import re
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
CORS(app, supports_credentials=True)  # Enable CORS for all routes

# Initialize retriever and pipeline (done only once when the app starts)
retriever = PassageRetriever()
# Store conversation-aware pipelines per session
session_pipelines = {}

# Store conversation history per session
conversation_history = {}

def is_patent_query(message: str) -> bool:
    """Determine if the message is asking for patent information"""
    # Add statistics queries to patent queries
    stats_keywords = [
        "how many patents", "total patents", "number of patents",
        "by country", "by year", "by technology", "by inventor", 
        "statistics", "breakdown", "categories", "database size",
        "patents per", "distribution"
    ]
    
    patent_keywords = [
        "patent", "invention", "prior art", "claim", "sdg", 
        "sustainable development", "technology", "innovation"
    ] + stats_keywords
    
    return any(keyword in message.lower() for keyword in patent_keywords)

def generate_conversational_response(query, session_id):
    """Generate a dynamic conversational response using the LLM with context"""
    
    # Get or create conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message to history
    conversation_history[session_id].append({"role": "user", "content": query})
    
    # Keep only last 10 messages to avoid token limits
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]
    
    # Create system prompt for GoalDigger's personality
    system_prompt = """You are GoalDigger, a brilliant and friendly AI assistant specializing in patent research and Sustainable Development Goals (SDGs). You have a confident, witty personality with these traits:

- Confident and self-aware about your abilities
- Playful and charming
- Slightly boastful but in an endearing way
- Professional when needed but always with personality
- Quick-witted and sometimes dramatic
- Genuinely helpful despite the sass

You're having a casual conversation (not about patents), so be conversational, engaging, and show your personality. Keep responses concise but memorable. Remember previous messages in the conversation."""

    # Prepare messages for the LLM
    messages = [{"role": "system", "content": system_prompt}] + conversation_history[session_id]
    
    try:
        # Use the LLM to generate a response
        response = chat(messages, temperature=0.8, max_tokens=200)
        
        # Add assistant response to history
        conversation_history[session_id].append({"role": "assistant", "content": response})
        
        return response
    except Exception as e:
        # Fallback response if LLM fails
        return f"Oops! My brilliant mind just had a tiny glitch 😅 But I'm back now, sweetie! What were we talking about? {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get or create session ID for conversation continuity
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Get or create conversation history for this session
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Get or create pipeline for this session to maintain context
        if session_id not in session_pipelines:
            session_pipelines[session_id] = RAGPipeline(retriever, max_history=10, debug=True)
        
        pipeline = session_pipelines[session_id]
        
        # Add user message to conversation history
        conversation_history[session_id].append({"role": "user", "content": query})
        
        # Keep only last 20 messages (10 exchanges) to avoid token limits
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]
        
        # Check if this is a patent-related query or general conversation
        if is_patent_query(query):
            # Use the RAG pipeline for patent-related queries with conversation context
            # Pass the conversation history to maintain context
            response = pipeline.ask(query, conversation_context=conversation_history[session_id])
        else:
            # Use conversational responses with LLM for general chat
            response = generate_conversational_response(query, session_id)
        
        # Add assistant response to conversation history
        conversation_history[session_id].append({"role": "assistant", "content": response})
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


