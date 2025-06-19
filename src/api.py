from flask import Flask, request, jsonify, session
from flask_cors import CORS
import logging
import os
import sys
import re
import json
import uuid
from datetime import timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Add logger
logger = logging.getLogger(__name__)

# Import the main components
try:
    from sql_retriever import SQLRetriever
    from llm_clients import chat
    from stats_queries import PatentStatistics
    from query_classifier import QueryClassifier
    from config import BACKEND_PORT, FRONTEND_PORT
    from patent_utils import format_patent_number, remove_similarity_from_text
    from queries.query_manager import QueryManager
except ImportError:
    from src.sql_retriever import SQLRetriever
    from src.llm_clients import chat
    from src.stats_queries import PatentStatistics
    from src.query_classifier import QueryClassifier
    from src.config import BACKEND_PORT, FRONTEND_PORT
    from src.patent_utils import format_patent_number, remove_similarity_from_text
    from src.queries.query_manager import QueryManager

# Type definitions for better code organization
@dataclass
class ConversationContext:
    latest_non_query: Optional[str] = None
    query_contexts: List[str] = None
    recent_responses: List[str] = None
    user_topics: List[str] = None

    def __post_init__(self):
        self.query_contexts = self.query_contexts or []
        self.recent_responses = self.recent_responses or []
        self.user_topics = self.user_topics or []

@dataclass
class ApiResponse:
    message: str
    chart: Optional[Dict] = None
    error: Optional[str] = None
    insight: Optional[str] = None
    takeaway: Optional[str] = None

# Initialize Flask app and configure CORS
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.permanent_session_lifetime = timedelta(hours=1)

cors_origins = [
    f"http://localhost:{FRONTEND_PORT}",
    "http://localhost:3000",
    "http://localhost:3001"
]
CORS(app, supports_credentials=True, origins=cors_origins)

# Initialize components
retriever = SQLRetriever()
classifier = QueryClassifier()
stats = PatentStatistics()
query_manager = QueryManager()

# Global state
session_pipelines = {}
conversation_history = {}

def get_context_from_history(conversation_history: list, max_messages: int = 10) -> dict:
    """Extract context from conversation history"""
    if not conversation_history:
        return {
            "latest_non_query": None,
            "query_contexts": [],
            "recent_responses": [],
            "user_topics": []
        }
    
    recent_messages = conversation_history[-max_messages:]
    latest_non_query = None
    query_contexts = []
    recent_responses = []
    user_topics = set()
    
    # Collect recent assistant responses
    for i, message in enumerate(recent_messages):
        if message.get("role") == "assistant":
            if i > 0 and recent_messages[i-1].get("role") == "user":
                user_msg = recent_messages[i-1].get("content", "").strip()
                ai_msg = message.get("content", "").strip()
                if user_msg and ai_msg:
                    context_pair = f"User asked: '{user_msg[:50]}...' and I responded with: '{ai_msg[:100]}...'"
                    recent_responses.append(context_pair)
    
    # Process user messages
    for message in reversed(recent_messages):
        if message.get("role") == "user":
            user_message = message.get("content", "").strip()
            
            # Extract potential topics
            words = user_message.lower().split()
            for word in words:
                if len(word) > 4 and word not in ["about", "what", "which", "where", "when", "there", "their", "these", "those"]:
                    user_topics.add(word)
            
            # Check for query indicators
            query_indicators = ["find", "search", "show", "how many", "what", "which", "patent", "statistics", "tell me about", "explain"]
            is_likely_query = any(indicator in user_message.lower() for indicator in query_indicators)
            
            if is_likely_query:
                if len(query_contexts) < 5:
                    query_contexts.append(user_message)
            elif latest_non_query is None:
                latest_non_query = user_message
    
    return {
        "latest_non_query": latest_non_query,
        "query_contexts": list(reversed(query_contexts)),
        "recent_responses": recent_responses[-3:],
        "user_topics": list(user_topics)[:10]
    }

def handle_search(query: str, context: ConversationContext) -> ApiResponse:
    """Handle patent search queries - regular semantic search or SQL-first for patent numbers"""
    try:
        # Check if this is a patent number search (just the number, not a detail request)
        patent_number = None
        import re
        patent_patterns = [
            r'\b(EP\d{7}[A-Z]\d?)\b',  
            r'\b(US\d{7,8}[A-Z]?\d?)\b',  
            r'\b(WO\d{4}/\d{6}[A-Z]?\d?)\b',  
            r'\b([A-Z]{2}\d{7,8}[A-Z]?\d?)\b',
            r'\b(\d{1,8})\b',  # Simple numeric patent numbers
        ]
        
        # Only treat as patent number search if it's JUST a patent number, not a detail request
        query_lower = query.lower().strip()
        detail_indicators = [
            "details of", "detail of", "details about", "detail about", 
            "information about", "tell me about", "show me details",
            "show me information", "what is", "describe", "explain", 
            "about patent", "patent details", "patent information",
            "more about", "full details", "complete information",
            "more details", "more detail", "detailed information",
            "get detailed", "show detailed", "comprehensive information",
            "claims for", "patent claims", "detailed claims",
            "description of", "detailed description", "technical details"
        ]
        
        has_detail_request = any(indicator in query_lower for indicator in detail_indicators)
        
        # Only do SQL-first search if it's a patent number WITHOUT detail request indicators
        if not has_detail_request:
            for pattern in patent_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    # Check if this is mostly just a patent number
                    patent_number = match.group(1).upper()
                    query_without_patent = query_lower.replace(patent_number.lower(), "").strip()
                    remaining_words = [word for word in query_without_patent.split() if len(word) > 2]
                    
                    # If there are fewer than 2 meaningful words left, treat as patent number search
                    if len(remaining_words) < 2:
                        return _handle_sql_first_patent_search(patent_number, query, context)
                    break
        
        # Regular semantic search for all other queries (technology searches, complex queries, etc.)
        results = retriever.search(query)
        if not results:
            return ApiResponse(
                message="I couldn't find any patents matching your query. Try rephrasing your search or using different keywords."
            )

        # Process results in batches
        filtered_results = results[:10]
        response = _process_search_results(query, filtered_results, context)
        return ApiResponse(message=remove_similarity_from_text(response) if response else "No response generated.")
    except Exception as e:
        return ApiResponse(
            message="I encountered an error while searching patents.",
            error=str(e)
        )

def handle_stats(query: str, context: ConversationContext = None) -> ApiResponse:
    """Handle statistical queries about patent data using the new modular system"""
    try:
        # First try the new modular query system
        query_response = query_manager.route_query(query)
        
        if query_response and query_response.message:
            return ApiResponse(
                message=query_response.message,
                chart=query_response.chart,
                insight=getattr(query_response, 'insight', ''),
                takeaway=getattr(query_response, 'takeaway', '')
            )
        
        # Fallback to legacy handlers for queries not covered by modular system
        query_lower = query.lower()
        
        # Map query types to handler functions
        handlers = {
            "basic": (_handle_basic_stats, ["how many patents", "total patents", "database size"]),
            "sdg_trends": (_handle_sdg_trends, ["sdg trend", "sdg over time", "sdg yearly"]),
            "tech": (_handle_tech_fields, ["technology field", "tech field"])
        }

        # Find matching handler
        for handler_type, (handler_func, keywords) in handlers.items():
            if any(keyword in query_lower for keyword in keywords):
                try:
                    return handler_func()
                except Exception as e:
                    return ApiResponse(
                        message=f"I encountered an error while generating {handler_type} statistics.",
                        error=str(e)
                    )

        return ApiResponse(
            message="I couldn't determine what type of statistics you're looking for. Please specify if you want to know about patent counts, trends, SDGs, technology analysis, inventors, or geographical information."
        )
    except Exception as e:
        return ApiResponse(
            message="I encountered an error while generating statistics.",
            error=str(e)
        )

def handle_conversation(query: str, context: ConversationContext) -> ApiResponse:
    """Handle conversational queries"""
    try:
        messages = _build_conversation_messages(query, context)
        response = chat(messages)
        return ApiResponse(message=response)
    except Exception as e:
        return ApiResponse(
            message="I encountered an error in our conversation.",
            error=str(e)
        )

def handle_patent_detail(query: str, patent_number: str, context: ConversationContext) -> ApiResponse:
    """Handle patent detail lookup queries with comprehensive embedded data"""
    try:
        # Use the comprehensive embedded data search for detailed requests
        return _get_detailed_patent_claims_and_description(patent_number, query, context)
            
    except Exception as e:
        return ApiResponse(
            message=f"I encountered an error while looking up detailed information for patent {patent_number}. Please try again.",
            error=str(e)
        )

def _get_semantic_patent_info(patent_number: str, patent_title: str) -> Optional[str]:
    """Get additional patent information from semantic search"""
    try:
        # Use the existing retriever to search for information about this patent
        search_query = f"{patent_number} {patent_title}" if patent_title else patent_number
        
        # Search for related chunks with higher k for more comprehensive results
        chunks = retriever.search(search_query, top_k_return=10)
        
        if chunks:
            # Extract relevant information from the chunks
            relevant_info = []
            technical_details = []
            applications = []
            
            for chunk in chunks:
                content = chunk.get('text', '').strip()  # Use 'text' field from retriever result
                content_lower = content.lower()
                
                # Check if this chunk is about the specific patent
                is_relevant = (
                    patent_number.upper() in content.upper() or 
                    (patent_title and any(word in content_lower for word in patent_title.lower().split() if len(word) > 3))
                )
                
                if is_relevant:
                    # Categorize the content based on keywords
                    if any(keyword in content_lower for keyword in ['application', 'use', 'industry', 'field']):
                        # This looks like application/usage information
                        if len(content) > 300:
                            content = content[:300] + "..."
                        applications.append(content)
                    elif any(keyword in content_lower for keyword in ['technical', 'method', 'process', 'invention', 'claim']):
                        # This looks like technical details
                        if len(content) > 350:
                            content = content[:350] + "..."
                        technical_details.append(content)
                    else:
                        # General relevant information
                        if len(content) > 250:
                            content = content[:250] + "..."
                        relevant_info.append(content)
            
            # Build comprehensive response
            response_parts = []
            
            if technical_details:
                response_parts.append("**Technical Details:**")
                for detail in technical_details[:2]:  # Limit to top 2
                    response_parts.append(f"• {detail}")
                response_parts.append("")
            
            if applications:
                response_parts.append("**Applications & Uses:**")
                for app in applications[:2]:  # Limit to top 2
                    response_parts.append(f"• {app}")
                response_parts.append("")
            
            if relevant_info:
                response_parts.append("**Additional Information:**")
                for info in relevant_info[:2]:  # Limit to top 2
                    response_parts.append(f"• {info}")
            
            if response_parts:
                return "\n".join(response_parts)
        
        return None
    except Exception as e:
        logger.warning(f"Error getting semantic patent info: {e}")
        return None

def _get_broader_patent_context(title: str, abstract: str) -> Optional[str]:
    """Get broader context about patent technology area when specific info not found"""
    try:
        if not title and not abstract:
            return None
            
        # Extract key terms from title and abstract
        key_terms = []
        if title:
            # Extract meaningful words from title (skip common words)
            title_words = [word for word in title.lower().split() if len(word) > 3 and word not in ['patent', 'method', 'system', 'apparatus']]
            key_terms.extend(title_words[:3])  # Top 3 words from title
        
        if abstract:
            # Extract a few key phrases from abstract
            abstract_words = [word for word in abstract.lower().split() if len(word) > 4]
            key_terms.extend(abstract_words[:5])  # Top 5 words from abstract
        
        if key_terms:
            # Search using key terms
            search_query = " ".join(key_terms[:6])  # Limit search terms
            chunks = retriever.search(search_query, top_k_return=5)
            
            if chunks:
                relevant_info = []
                for chunk in chunks:
                    content = chunk.get('text', '').strip()  # Use 'text' field from retriever result
                    if len(content) > 200:
                        content = content[:200] + "..."
                    relevant_info.append(content)
                
                if relevant_info:
                    return "**Related Technology Context:**\n" + "\n".join(f"• {info}" for info in relevant_info[:3])
        
        return None
    except Exception as e:
        logger.warning(f"Error getting broader patent context: {e}")
        return None

def _format_patent_details(patent, requested_number: str) -> str:
    """Format patent details into a readable response"""
    import json
    from datetime import datetime
    
    # Format the complete patent number
    country = patent.publication_country or ""
    number = patent.publication_number or ""
    kind = patent.publication_kind or ""
    
    # Create properly formatted patent number
    if country and number:
        if kind:
            formatted_number = f"{country}{number.zfill(7)}{kind}"
        else:
            formatted_number = f"{country}{number.zfill(7)}"
    else:
        formatted_number = number
    
    # Basic patent information
    title = patent.title_en or "Title not available"
    
    # Publication date formatting
    pub_date = "Date not available"
    if patent.publication_date:
        try:
            if isinstance(patent.publication_date, str):
                date_obj = datetime.strptime(patent.publication_date, "%Y-%m-%d")
            else:
                date_obj = patent.publication_date
            pub_date = date_obj.strftime("%B %d, %Y")
        except:
            pub_date = str(patent.publication_date)
    
    response = f"# Patent Details: {formatted_number}\n\n"
    response += f"**Title:** {title}\n\n"
    response += f"**Publication Date:** {pub_date}\n\n"
    
    # Abstract
    if patent.abstract_text:
        abstract = patent.abstract_text.strip()
        # Show more of the abstract - up to 1500 characters instead of 500
        if len(abstract) > 1500:
            abstract = abstract[:1500] + "..."
        response += f"**Abstract:**\n{abstract}\n\n"
    
    # Inventors and Applicants
    try:
        if patent.inventor_names:
            inventors = json.loads(patent.inventor_names) if isinstance(patent.inventor_names, str) else patent.inventor_names
            if inventors and isinstance(inventors, list):
                response += f"**Inventors:** {', '.join(inventors[:5])}"
                if len(inventors) > 5:
                    response += f" (and {len(inventors) - 5} more)"
                response += "\n\n"
        
        if patent.applicant_names:
            applicants = json.loads(patent.applicant_names) if isinstance(patent.applicant_names, str) else patent.applicant_names
            if applicants and isinstance(applicants, list):
                response += f"**Applicants:** {', '.join(applicants[:3])}"
                if len(applicants) > 3:
                    response += f" (and {len(applicants) - 3} more)"
                response += "\n\n"
    except:
        pass  # Skip if JSON parsing fails
    
    # Countries
    try:
        if patent.applicant_countries:
            countries = json.loads(patent.applicant_countries) if isinstance(patent.applicant_countries, str) else patent.applicant_countries
            if countries and isinstance(countries, list):
                response += f"**Countries:** {', '.join(set(countries))}\n\n"
    except:
        pass
    
    # IPC Classification
    try:
        if patent.ipc:
            ipc_data = json.loads(patent.ipc) if isinstance(patent.ipc, str) else patent.ipc
            if ipc_data and isinstance(ipc_data, list):
                # Show all IPC codes instead of limiting to 5
                response += f"**IPC Classification:** {', '.join(ipc_data)}\n\n"
    except:
        pass
    
    # Technology fields
    try:
        if patent.ipc_technologies:
            tech_data = json.loads(patent.ipc_technologies) if isinstance(patent.ipc_technologies, str) else patent.ipc_technologies
            if tech_data and isinstance(tech_data, list):
                response += f"**Technology Areas:** {', '.join(tech_data[:5])}"
                if len(tech_data) > 5:
                    response += f" (and {len(tech_data) - 5} more)"
                response += "\n\n"
    except:
        pass
    
    # SDG information
    try:
        if patent.sdg_number:
            sdg_data = json.loads(patent.sdg_number) if isinstance(patent.sdg_number, str) else patent.sdg_number
            if sdg_data and isinstance(sdg_data, list):
                sdg_list = [f"SDG {num}" for num in sdg_data if isinstance(num, int)]
                if sdg_list:
                    response += f"**Related UN SDGs:** {', '.join(sdg_list)}\n\n"
    except:
        pass
    
    # Analysis and explanations
    if patent.analysis_explanation:
        try:
            analysis = json.loads(patent.analysis_explanation) if isinstance(patent.analysis_explanation, str) else patent.analysis_explanation
            
            # Handle different data structures for analysis
            analysis_text = ""
            if isinstance(analysis, str):
                analysis_text = analysis.strip()
            elif isinstance(analysis, list):
                # If it's a list, join the items properly
                formatted_items = []
                for item in analysis:
                    if isinstance(item, str):
                        # Clean up the item and format it properly
                        clean_item = item.strip()
                        if clean_item:
                            formatted_items.append(clean_item)
                if formatted_items:
                    analysis_text = "\n".join(f"• {item}" for item in formatted_items)
            elif isinstance(analysis, dict) and 'text' in analysis:
                analysis_text = analysis['text'].strip()
            else:
                analysis_text = str(analysis).strip()
            
            if analysis_text and len(analysis_text) > 50:
                # Don't truncate technical analysis - show it all
                response += f"**Technical Analysis:**\n{analysis_text}\n\n"
        except Exception as e:
            # If there's an error, show what we can
            if patent.analysis_explanation:
                response += f"**Technical Analysis:**\n{str(patent.analysis_explanation)}\n\n"
    
    response += "---\n\n"
    response += "💡 **Need more information?** You can ask me to:\n"
    response += "• Find related patents in the same technology area\n"
    response += "• Show patents by the same inventors or applicants\n"
    response += "• Search for patents citing this one\n"
    response += "• Analyze patent trends in this field"
    
    return response

def _process_search_results(query: str, results: List[Dict], context: ConversationContext) -> str:
    """Process search results and generate response"""
    batches = _batch_results(results)
    
    if len(batches) == 1:
        return _generate_batch_response(query, batches[0], context)
    
    # Process multiple batches
    batch_responses = []
    for i, batch in enumerate(batches, 1):
        response = _generate_batch_response(
            f"{query} (analyzing batch {i}/{len(batches)})", 
            batch,
            context
        )
        batch_responses.append(f"**Analysis Part {i}:**\n{response}")
    
    # Combine and summarize
    combined_response = "\n\n".join(batch_responses)
    summary = _generate_summary(query, combined_response, len(results))
    return f"{summary}\n\n---\n\n{combined_response}"

def _batch_results(results: List[Dict], max_tokens: int = 12000) -> List[List[Dict]]:
    """Batch results to fit within token limits - increased for comprehensive responses"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for result in results:
        result_text = result.get('text', '') + result.get('title', '') + result.get('abstract', '')
        result_tokens = len(result_text) // 4  # Rough estimation
        
        if current_tokens + result_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [result]
            current_tokens = result_tokens
        else:
            current_batch.append(result)
            current_tokens += result_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def _generate_batch_response(query: str, batch: List[Dict], context: ConversationContext) -> str:
    """Generate response for a batch of results with increased token limit"""
    prompt = _build_search_prompt(query, batch, context)
    try:
        return chat([{"role": "user", "content": prompt}], max_tokens=16384)  # Increased token limit
    except Exception:
        return _generate_fallback_response(batch)

def _generate_summary(query: str, analysis: str, result_count: int) -> str:
    """Generate summary for multiple batches of results"""
    prompt = f"""Based on the following analyses of {result_count} patents related to "{query}", provide a comprehensive summary:

{analysis}

Please provide a detailed summary highlighting:
1. Key findings and main themes
2. Most relevant patents with their numbers and brief descriptions
3. Technical insights and innovations
4. Potential applications and implications
5. Notable inventors or organizations

Be comprehensive but well-organized."""
    
    try:
        return chat([{"role": "user", "content": prompt}], max_tokens=16384)  # Increased token limit
    except Exception:
        return f"Analysis of {result_count} Patents:"

def _build_search_prompt(query: str, results: List[Dict], context: ConversationContext) -> str:
    """Build comprehensive prompt for search results"""
    context_parts = []
    if context.latest_non_query:
        context_parts.append(f"User context: {context.latest_non_query}")
    if context.query_contexts:
        context_parts.append(f"Previous related queries: {'; '.join(context.query_contexts)}")
    
    prompt = [
        "You are a patent research assistant specialized in providing comprehensive, detailed analysis.",
        "Analyze the following patent information and provide a thorough, well-structured response to the user's query.",
        "Include specific patent numbers, technical details, applications, and insights.",
        "Organize your response with clear sections and provide actionable information.",
        "",
        f"Query: {query}",
        "",
    ]
    
    if context_parts:
        prompt.extend(context_parts)
        prompt.append("")
    
    prompt.append("Patent Data:")
    
    for i, result in enumerate(results, 1):
        similarity = result.get('similarity', 0)
        formatted_number = format_patent_number(
            result.get('publication_country'),
            result.get('publication_number', 'N/A'),
            result.get('publication_kind')
        )
        
        prompt.extend([
            f"\nPatent {i} - Similarity: {similarity:.3f}{_get_relevance_note(similarity)}:",
            f"- Publication Number: {formatted_number}",
            f"- Title: {result.get('title', 'No title available')}",
        ])
        
        # Include more text for comprehensive analysis
        text_content = result.get('text', '')
        if text_content:
            text_preview = text_content[:800] + "..." if len(text_content) > 800 else text_content
            prompt.append(f"- Content: {text_preview}")
        
        if result.get('abstract'):
            abstract_preview = result['abstract'][:400] + "..." if len(result['abstract']) > 400 else result['abstract']
            prompt.append(f"- Abstract: {abstract_preview}")
        
        # Add additional fields if available
        if result.get('inventor_names'):
            prompt.append(f"- Inventors: {result['inventor_names']}")
        if result.get('assignee_organization'):
            prompt.append(f"- Assignee: {result['assignee_organization']}")
        if result.get('ipc_codes'):
            prompt.append(f"- IPC Classification: {result['ipc_codes']}")
    
    prompt.extend([
        "",
        "INSTRUCTIONS:",
        "- Provide a comprehensive, well-structured analysis",
        "- Include specific patent numbers and technical details",
        "- Organize information by themes or categories when relevant",
        "- Highlight key innovations, applications, and technical approaches", 
        "- Focus on the most relevant patents but mention others briefly",
        "- Use clear headings and bullet points for readability",
        "- Do not include similarity scores in your response",
        "- Provide actionable insights and technical depth"
    ])
    
    return "\n".join(prompt)

def _get_relevance_note(similarity: float) -> str:
    """Get relevance note based on similarity score"""
    if similarity > 0.6:
        return " (High relevance)"
    elif similarity > 0.3:
        return " (Medium relevance - analyze critically)"
    return " (Low relevance - be skeptical)"

def _generate_fallback_response(results: List[Dict]) -> str:
    """Generate fallback response if LLM fails"""
    response_parts = [f"I found {len(results)} relevant patents:"]
    
    for i, result in enumerate(results, 1):
        formatted_number = format_patent_number(
            result.get('publication_country'),
            result.get('publication_number', 'N/A'),
            result.get('publication_kind')
        )
        
        response_parts.extend([
            f"\n{i}. **Patent {formatted_number}**",
            f"   Title: {result.get('title', 'No title available')}",
            f"   Extract: {result.get('text', '')[:200]}..."
        ])
    
    return "\n".join(response_parts)

def _build_conversation_messages(query: str, context: ConversationContext) -> List[Dict]:
    """Build messages for conversation"""
    messages = [{
        "role": "system",
        "content": _get_system_prompt()
    }]
    
    if context.latest_non_query:
        messages.extend([
            {"role": "user", "content": f"Previous context: {context.latest_non_query}"},
            {"role": "assistant", "content": "I'll keep our previous conversation in mind."}
        ])
    
    if context.query_contexts:
        query_context_str = "\n".join([f"- {q}" for q in context.query_contexts[-3:]])
        messages.extend([
            {"role": "user", "content": f"Recent topics:\n{query_context_str}"},
            {"role": "assistant", "content": "I'll consider our discussion history."}
        ])
    
    messages.append({"role": "user", "content": query})
    return messages

def _get_system_prompt() -> str:
    """Get system prompt for conversation"""
    return """You are GoalDigger, a helpful AI assistant specialized in patents and SDGs.
Maintain a friendly, professional tone and natural conversation flow.
Connect information to previous context when relevant.
Explain technical terms naturally within responses.
Use concrete examples to illustrate complex points."""

# Statistical response handlers
def _handle_basic_stats() -> ApiResponse:
    """Handle basic statistics query"""
    data = stats.get_basic_stats()
    if not data:
        return ApiResponse(message="Sorry, I couldn't retrieve the basic statistics.")
    
    response = [
        "**Database Overview:**\n",
        f"📊 Total Patents: {data['total_patents']:,}",
        f"📄 Total Chunks: {data['total_chunks']:,}"
    ]
    
    if data['date_range']['earliest'] and data['date_range']['latest']:
        response.append(
            f"📅 Date Range: {data['date_range']['earliest']} to {data['date_range']['latest']}"
        )
    
    response.append("\n🌍 **Top Countries by Patent Count:**")
    for country in data['top_countries'][:5]:
        response.append(f"  • {country['country']}: {country['count']:,} patents")
    
    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_publication_trends(query: str = "") -> ApiResponse:
    """Handle publication trends query with enhanced functionality"""
    from src.api_enhanced import _handle_publication_trends_enhanced
    
    try:
        result = _handle_publication_trends_enhanced(query)
        return ApiResponse(
            message=result['message'],
            chart=result['chart']
        )
    except Exception as e:
        print(f"Error in enhanced publication trends: {e}")
        return ApiResponse(message="Sorry, I couldn't retrieve the publication trends.")
    
    # Show appropriate data based on request
    if months and data['monthly']:
        response.append("📈 **Patents by Month:**")
        # Show monthly data
        display_months = data['monthly'][-months:] if months <= 24 else data['monthly']
        for month_data in display_months:
            year, month, count = month_data['year'], month_data['month'], month_data['count']
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]
            response.append(f"  • {month_name} {year}: {count:,} patents")
        
        total_monthly = sum(month_data['count'] for month_data in display_months)
        response.append(f"\n📅 **{time_period.title()} Total:** {total_monthly:,} patents")
        
    elif data['yearly']:
        response.append("� **Patents by Year:**")
        # Show yearly data
        display_years = data['yearly'][-years:] if years <= 10 else data['yearly']
        for year in display_years:
            response.append(f"  • {year['year']}: {year['count']:,} patents")
        
        if data['monthly']:
            total_last_year = sum(month['count'] for month in data['monthly'])
            response.append(f"\n📅 **Last 12 Months Total:** {total_last_year:,} patents")

    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_sdg_stats() -> ApiResponse:
    """Handle SDG statistics query"""
    data = stats.get_sdg_distribution()
    if not data:
        return ApiResponse(message="Sorry, I couldn't retrieve the SDG distribution.")
    
    distribution = data.get('distribution', data)
    total_sdg_patents = sum(item['count'] for item in distribution)
    
    response = [
        "**SDG Distribution:**\n",
        f"📊 Total Patents with SDG Classification: {total_sdg_patents:,}\n",
        "🎯 **Top SDG Categories:**"
    ]
    
    for sdg in distribution[:10]:
        response.append(f"  • SDG {sdg['sdg']}: {sdg['count']:,} patents")
    
    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_sdg_trends() -> ApiResponse:
    """Handle SDG trends query"""
    data = stats.get_sdg_trends_over_time(10)
    if not data:
        return ApiResponse(message="Sorry, I couldn't retrieve the SDG trends.")
    
    response = [
        "**SDG Patent Trends Over Time:**\n",
        f"📅 Time Period: {data['years_covered']}",
        f"📊 Total Patents with SDG Classification: {data['total_patents']:,}\n",
        "🎯 **Top SDG Categories (Total Count):**"
    ]
    
    for sdg in data['top_sdgs'][:5]:
        response.append(f"  • SDG {sdg['sdg']}: {sdg['total_count']:,} patents")
    
    if data['yearly_trends']:
        response.extend([
            "\n📈 **Recent Yearly Trends:**",
            *(f"  • {year['year']}: {sum(v for k, v in year.items() if k != 'year'):,} SDG-classified patents"
             for year in data['yearly_trends'][-5:])
        ])
    
    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_tech_fields() -> ApiResponse:
    """Handle technology fields query"""
    data = stats.get_technology_fields()
    if not data:
        return ApiResponse(message="Sorry, I couldn't retrieve the technology fields distribution.")
    
    distribution = data.get('distribution', data)
    response = [
        "**Technology Fields Distribution:**\n",
        "🔬 **Top Technology Fields:**"
    ]
    
    for tech in distribution[:10]:
        response.append(f"  • {tech['field']}: {tech['count']:,} patents")
    
    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_patent_coverage(query: str) -> ApiResponse:
    """Handle patent coverage query"""
    # Extract patent number from query
    patent_number_match = re.search(r'patent\s+([A-Za-z]{1,3}\d+\w*)', query, re.IGNORECASE)
    if not patent_number_match:
        return ApiResponse(message="Sorry, I couldn't identify a patent number in your query. Please provide a patent number like 'US123456' or 'EP1234567'.")
    
    patent_number = patent_number_match.group(1).strip().upper()
    
    # Get patent coverage data
    data = stats.get_patent_coverage(patent_number)
    if not data:
        return ApiResponse(message=f"Sorry, I couldn't find patent {patent_number} in our database.")
    
    # Format the response
    active_count = data['active_count']
    extension_count = data['extension_count']
    active_duration = data['active_duration']['text']
    title = data['title'] or "Untitled Patent"
    
    response = [
        f"**Patent Coverage for {patent_number}:**\n",
        f"📄 **Title:** {title}",
        f"🌍 **Active in:** {active_count} {'country' if active_count == 1 else 'countries'}",
        f"🔄 **Extended to:** {extension_count} {'country' if extension_count == 1 else 'countries'}",
        f"⏱️ **Active for:** {active_duration}"
    ]
    
    # Add country details if available
    if data['active_countries'] and len(data['active_countries']) > 0:
        response.append("\n**Active Countries:**")
        response.append(", ".join(data['active_countries'][:10]))
        if len(data['active_countries']) > 10:
            response.append(f"... and {len(data['active_countries']) - 10} more")
    
    if data['extension_countries'] and len(data['extension_countries']) > 0:
        response.append("\n**Extension Countries:**")
        response.append(", ".join(data['extension_countries'][:10]))
        if len(data['extension_countries']) > 10:
            response.append(f"... and {len(data['extension_countries']) - 10} more")
    
    return ApiResponse(
        message="\n".join(response),
        chart=data.get('chart')
    )

def _handle_comprehensive_patent_info(patent_number: str, original_query: str, context: ConversationContext) -> ApiResponse:
    """
    Handle comprehensive patent information combining database lookup and semantic search.
    This provides the most complete information about a patent.
    """
    try:
        from database.config import get_db_session_simple
        from database.models import Patent
        
        response_parts = []
        
        # 1. Database lookup for structured information
        session = get_db_session_simple()
        patent = None
        
        try:
            # Try multiple variations to find the patent
            search_variations = [
                patent_number,
                patent_number.upper(),
                patent_number.replace('EP', '').replace('US', '').replace('WO', ''),
                re.sub(r'[A-Z]', '', patent_number),  # Remove letters
                patent_number.lstrip('0')  # Remove leading zeros
            ]
            
            for variation in search_variations:
                patent = session.query(Patent).filter(
                    Patent.publication_number == variation
                ).first()
                if patent:
                    break
            
            if patent:
                # Add structured database information
                response_parts.append("## Patent Information from Database")
                response_parts.append(f"**Patent Number:** {patent.publication_number}")
                response_parts.append(f"**Title:** {patent.title_en or 'Not available'}")
                
                if patent.abstract_text:
                    abstract_preview = patent.abstract_text[:500] + "..." if len(patent.abstract_text) > 500 else patent.abstract_text
                    response_parts.append(f"**Abstract:** {abstract_preview}")
                
                if patent.publication_date:
                    response_parts.append(f"**Publication Date:** {patent.publication_date}")
                
                if patent.ipc:
                    response_parts.append(f"**IPC Classification:** {patent.ipc}")
                
                if patent.inventor_names:
                    # inventor_names is a JSON string, need to parse it
                    try:
                        inventors = json.loads(patent.inventor_names) if isinstance(patent.inventor_names, str) else patent.inventor_names
                        response_parts.append(f"**Inventors:** {', '.join(inventors) if isinstance(inventors, list) else inventors}")
                    except:
                        response_parts.append(f"**Inventors:** {patent.inventor_names}")
                
                if patent.applicant_names:
                    # applicant_names is also a JSON string
                    try:
                        applicants = json.loads(patent.applicant_names) if isinstance(patent.applicant_names, str) else patent.applicant_names
                        response_parts.append(f"**Applicant:** {', '.join(applicants) if isinstance(applicants, list) else applicants}")
                    except:
                        response_parts.append(f"**Applicant:** {patent.applicant_names}")
                
                if patent.publication_country:
                    response_parts.append(f"**Country:** {patent.publication_country}")
                
                response_parts.append("")  # Empty line for separation
                
        finally:
            session.close()
        
        # 2. Semantic search for detailed content and context
        semantic_results = retriever.search(f"{patent_number}", top_k_return=20)  # Get more results for comprehensive info
        
        if semantic_results:
            response_parts.append("## Detailed Information from Patent Documents")
            
            # Categorize semantic results
            technical_content = []
            application_content = []
            claims_content = []
            background_content = []
            
            for chunk in semantic_results:
                content = chunk.get('text', '').strip()  # Use 'text' field from retriever result
                content_lower = content.lower()
                
                # Check if content is relevant to this patent
                is_relevant = patent_number.upper() in content.upper()
                
                if is_relevant or (patent and patent.title_en and 
                                 any(word in content_lower for word in patent.title_en.lower().split() if len(word) > 4)):
                    
                    # Categorize content
                    if any(keyword in content_lower for keyword in ['claim', 'claims', 'wherein', 'characterized']):
                        claims_content.append(content)
                    elif any(keyword in content_lower for keyword in ['technical', 'method', 'process', 'invention', 'mechanism']):
                        technical_content.append(content)
                    elif any(keyword in content_lower for keyword in ['application', 'use', 'industry', 'field', 'purpose']):
                        application_content.append(content)
                    elif any(keyword in content_lower for keyword in ['background', 'prior art', 'existing', 'conventional']):
                        background_content.append(content)
                    else:
                        technical_content.append(content)  # Default to technical
            
            # Add categorized content to response
            if technical_content:
                response_parts.append("### Technical Description")
                for content in technical_content[:3]:  # Top 3 most relevant
                    truncated = content[:800] + "..." if len(content) > 800 else content
                    response_parts.append(f"• {truncated}")
                response_parts.append("")
            
            if claims_content:
                response_parts.append("### Patent Claims")
                for content in claims_content[:2]:  # Top 2 claims
                    truncated = content[:600] + "..." if len(content) > 600 else content
                    response_parts.append(f"• {truncated}")
                response_parts.append("")
            
            if application_content:
                response_parts.append("### Applications & Uses")
                for content in application_content[:2]:  # Top 2 applications
                    truncated = content[:500] + "..." if len(content) > 500 else content
                    response_parts.append(f"• {truncated}")
                response_parts.append("")
            
            if background_content:
                response_parts.append("### Background & Context")
                for content in background_content[:2]:  # Top 2 background info
                    truncated = content[:400] + "..." if len(content) > 400 else content
                    response_parts.append(f"• {truncated}")
                response_parts.append("")
        
        # 3. If we found patent info, also search for related technologies
        if patent and patent.title_en:
            response_parts.append("## Related Technology Context")
            
            # Extract key technology terms
            title_words = [word for word in patent.title_en.lower().split() 
                          if len(word) > 4 and word not in ['patent', 'method', 'system', 'apparatus', 'device']]
            
            if title_words:
                tech_query = " ".join(title_words[:4])  # Use top 4 technical terms
                related_results = retriever.search(tech_query, top_k_return=8)
                
                tech_insights = []
                for chunk in related_results:
                    content = chunk.get('text', '').strip()  # Use 'text' field from retriever result
                    # Skip if it's about the same patent
                    if patent_number.upper() not in content.upper():
                        truncated = content[:300] + "..." if len(content) > 300 else content
                        tech_insights.append(truncated)
                
                if tech_insights:
                    response_parts.append("### Related Technology Insights")
                    for insight in tech_insights[:3]:  # Top 3 related insights
                        response_parts.append(f"• {insight}")
        
        # 4. Generate final response
        if response_parts:
            final_response = "\n".join(response_parts)
            
            # Use LLM to enhance and summarize if the response is very long
            if len(final_response) > 3000:
                try:
                    enhancement_prompt = f"""
                    Please organize and enhance this comprehensive patent information to make it more readable and insightful. 
                    Keep all the important technical details but organize them better and add brief explanatory context where helpful.
                    
                    Patent Information:
                    {final_response}
                    
                    Please provide a well-organized, comprehensive response about this patent.
                    """
                    
                    enhanced_response = chat([{"role": "user", "content": enhancement_prompt}], max_tokens=16384)
                    return ApiResponse(message=enhanced_response)
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}")
                    # Return the original comprehensive response if LLM enhancement fails
                    return ApiResponse(message=final_response)
            else:
                return ApiResponse(message=final_response)
        
        # If no information found
        return ApiResponse(
            message=f"I couldn't find comprehensive information about patent {patent_number}. "
                   f"The patent may not be in my database or the number might be incorrect. "
                   f"Please verify the patent number and try again."
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive patent info: {e}")
        return ApiResponse(
            message=f"I encountered an error while gathering comprehensive information about patent {patent_number}.",
            error=str(e)
        )

def _get_detailed_patent_claims_and_description(patent_number: str, original_query: str, context: ConversationContext) -> ApiResponse:
    """
    Get detailed patent claims and descriptions from embedded data.
    This is called when users specifically ask for more details about a patent.
    """
    try:
        from database.config import get_db_session_simple
        from database.models import Patent
        
        response_parts = []
        
        # First get basic SQL data for context
        session = get_db_session_simple()
        patent = None
        
        try:
            # Try to find the patent in SQL database for basic info
            search_variations = [
                patent_number,  # Original format
                patent_number.upper(),  # Uppercase
                patent_number.replace('EP', '').replace('US', '').replace('WO', ''),  # Remove prefix
                re.sub(r'[A-Z]', '', patent_number),  # Remove all letters
                # Better numeric extraction:
                re.search(r'\d+', patent_number).group() if re.search(r'\d+', patent_number) else None,  # Extract number
                patent_number.replace('EP', '').lstrip('0').rstrip('B1').rstrip('A1'),  # Clean format
            ]
            
            # Remove None values and duplicates
            search_variations = list(dict.fromkeys([v for v in search_variations if v]))
            
            for variation in search_variations:
                patent = session.query(Patent).filter(
                    Patent.publication_number == variation
                ).first()
                if patent:
                    break
                    
        finally:
            session.close()
        
        # Search embedded data for comprehensive patent content
        # Use multiple search variations to find the patent in embedded data
        search_queries = []
        
        # Add original patent number
        search_queries.append(patent_number)
        
        # If we found the patent in SQL, also search using the SQL patent number
        if patent:
            sql_patent_number = patent.publication_number
            if sql_patent_number != patent_number:
                search_queries.append(sql_patent_number)
            
            # Add title-based searches for better coverage
            if patent.title_en:
                title_words = patent.title_en.split()[:4]  # First 4 words of title
                search_queries.append(f"{sql_patent_number} {' '.join(title_words)}")
                search_queries.append(f"{patent_number} {' '.join(title_words)}")
        else:
            # If not found in SQL, try normalized variations
            normalized_variations = [
                patent_number.replace('EP', '').replace('US', '').replace('WO', ''),
                re.sub(r'[A-Z]', '', patent_number),
                patent_number.lstrip('0')
            ]
            search_queries.extend(normalized_variations)
        
        # Add specific search queries for different types of content
        for base_number in search_queries[:2]:  # Use top 2 variations to avoid too many queries
            search_queries.extend([
                f"claims {base_number}",
                f"description {base_number}",
                f"technical field {base_number}",
            ])
        
        # Collect all relevant embedded content
        all_relevant_content = []
        claims_content = []
        description_content = []
        technical_content = []
        background_content = []
        
        for search_query in search_queries:
            try:
                results = retriever.search(search_query, top_k_return=15)
                
                for result in results:
                    content = result.get('text', '').strip()
                    content_lower = content.lower()
                    
                    # Check if this content is specifically about our patent
                    # Use multiple matching criteria for better coverage
                    is_about_patent = False
                    
                    # Check for exact patent number match
                    if patent_number.upper() in content.upper():
                        is_about_patent = True
                    
                    # If we have SQL patent info, also check for SQL patent number
                    elif patent and patent.publication_number.upper() in content.upper():
                        is_about_patent = True
                    
                    # Check for title words match (if available)
                    elif patent and patent.title_en:
                        title_words = [word.lower() for word in patent.title_en.split() if len(word) > 4]
                        if len(title_words) >= 2:
                            matching_words = sum(1 for word in title_words if word in content_lower)
                            # If at least half the significant title words match, consider it relevant
                            if matching_words >= len(title_words) // 2:
                                is_about_patent = True
                    
                    # Check for normalized patent number variations
                    else:
                        normalized_patent = re.sub(r'[A-Z]', '', patent_number)
                        if normalized_patent.isdigit() and normalized_patent in content:
                            is_about_patent = True
                    
                    if is_about_patent and content not in [item['content'] for item in all_relevant_content]:
                        content_info = {
                            'content': content,
                            'similarity': result.get('similarity', 0)
                        }
                        all_relevant_content.append(content_info)
                        
                        # Categorize content
                        if any(keyword in content_lower for keyword in ['claim', 'claims', 'wherein', 'characterized by']):
                            claims_content.append(content_info)
                        elif any(keyword in content_lower for keyword in ['description', 'detailed description', 'embodiment', 'implementation']):
                            description_content.append(content_info)
                        elif any(keyword in content_lower for keyword in ['technical field', 'technical problem', 'technical solution', 'invention']):
                            technical_content.append(content_info)
                        elif any(keyword in content_lower for keyword in ['background', 'prior art', 'state of art', 'conventional']):
                            background_content.append(content_info)
                            
            except Exception as e:
                logger.warning(f"Error searching with query '{search_query}': {e}")
                continue
        
        # Sort all content by similarity
        all_relevant_content.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Build comprehensive response
        if patent:
            response_parts.append(f"# Detailed Information for Patent {patent.publication_number}")
            response_parts.append(f"**Title:** {patent.title_en}")
            response_parts.append("")
        else:
            response_parts.append(f"# Detailed Information for Patent {patent_number}")
            response_parts.append("")
        
        # Add patent claims section
        if claims_content:
            response_parts.append("## Patent Claims")
            response_parts.append("*Direct excerpts from the patent document:*")
            response_parts.append("")
            
            # Show top 3 most relevant claims
            for i, claim_info in enumerate(claims_content[:3], 1):
                content = claim_info['content']
                # Truncate very long claims but keep substantial content
                if len(content) > 1500:
                    content = content[:1500] + "...\n*[Claim continues]*"
                response_parts.append(f"**Claim Section {i}:**")
                response_parts.append(content)
                response_parts.append("")
        
        # Add detailed description section
        if description_content:
            response_parts.append("## Detailed Description")
            response_parts.append("*Technical implementation and embodiments:*")
            response_parts.append("")
            
            for i, desc_info in enumerate(description_content[:3], 1):
                content = desc_info['content']
                if len(content) > 1200:
                    content = content[:1200] + "...\n*[Description continues]*"
                response_parts.append(f"**Description Section {i}:**")
                response_parts.append(content)
                response_parts.append("")
        
        # Add technical field section
        if technical_content:
            response_parts.append("## Technical Field & Innovation")
            response_parts.append("*Technical problem and solution:*")
            response_parts.append("")
            
            for i, tech_info in enumerate(technical_content[:2], 1):
                content = tech_info['content']
                if len(content) > 800:
                    content = content[:800] + "...\n*[Technical details continue]*"
                response_parts.append(f"**Technical Aspect {i}:**")
                response_parts.append(content)
                response_parts.append("")
        
        # Add background context if available
        if background_content:
            response_parts.append("## Background & Prior Art")
            response_parts.append("*Context and existing technology:*")
            response_parts.append("")
            
            for i, bg_info in enumerate(background_content[:2], 1):
                content = bg_info['content']
                if len(content) > 600:
                    content = content[:600] + "...\n*[Background continues]*"
                response_parts.append(f"**Background {i}:**")
                response_parts.append(content)
                response_parts.append("")
        
        # If we have other relevant content not categorized above
        other_content = [item for item in all_relevant_content 
                        if item not in claims_content + description_content + technical_content + background_content]
        
        if other_content:
            response_parts.append("## Additional Patent Information")
            response_parts.append("")
            
            for i, other_info in enumerate(other_content[:2], 1):
                content = other_info['content']
                if len(content) > 600:
                    content = content[:600] + "..."
                response_parts.append(f"**Additional Detail {i}:**")
                response_parts.append(content)
                response_parts.append("")
        
        # Add summary of what was found
        response_parts.append("---")
        response_parts.append("📊 **Summary of Retrieved Information:**")
        response_parts.append(f"• {len(claims_content)} patent claims sections")
        response_parts.append(f"• {len(description_content)} detailed description sections")
        response_parts.append(f"• {len(technical_content)} technical field sections")
        response_parts.append(f"• {len(background_content)} background sections")
        response_parts.append(f"• {len(other_content)} additional relevant sections")
        response_parts.append(f"• Total: {len(all_relevant_content)} relevant document sections analyzed")
        
        if len(all_relevant_content) == 0:
            return ApiResponse(
                message=f"I couldn't find detailed embedded information for patent {patent_number}. "
                       f"This patent may not have comprehensive document data in my embedded database. "
                       f"You can try searching for related patents or ask for basic information instead."
            )
        
        final_response = "\n".join(response_parts)
        
        # If the response is very long, use LLM to organize it better
        if len(final_response) > 4000:
            try:
                organization_prompt = f"""
                Please organize this comprehensive patent information into a well-structured, readable format.
                Keep all the technical details but improve the organization and add brief explanatory notes where helpful.
                Make it easier to understand while preserving all the detailed claims and descriptions.
                
                Patent Information:
                {final_response}
                
                Please provide a well-organized, comprehensive response about this patent's detailed content.
                """
                
                organized_response = chat([{"role": "user", "content": organization_prompt}], max_tokens=16384)
                return ApiResponse(message=organized_response)
                
            except Exception as e:
                logger.warning(f"LLM organization failed: {e}")
                # Return the original detailed response if LLM organization fails
                return ApiResponse(message=final_response)
        
        return ApiResponse(message=final_response)
        
    except Exception as e:
        logger.error(f"Error getting detailed patent information: {e}")
        return ApiResponse(
            message=f"I encountered an error while retrieving detailed information for patent {patent_number}.",
            error=str(e)
        )

def _handle_sql_first_patent_search(patent_number: str, original_query: str, context: ConversationContext) -> ApiResponse:
    """
    Handle patent number search with SQL-first approach, then offer detailed embedded data.
    """
    try:
        from database.config import get_db_session_simple
        from database.models import Patent
        
        # Normalize patent number for database search
        session = get_db_session_simple()
        patent = None
        
        try:
            # Try multiple variations to find the patent - improved logic
            search_variations = [
                patent_number,  # Original format
                patent_number.upper(),  # Uppercase
                patent_number.replace('EP', '').replace('US', '').replace('WO', ''),  # Remove prefix
                re.sub(r'[A-Z]', '', patent_number),  # Remove all letters
                # Better numeric extraction:
                re.search(r'\d+', patent_number).group() if re.search(r'\d+', patent_number) else None,  # Extract number
                patent_number.replace('EP', '').lstrip('0').rstrip('B1').rstrip('A1'),  # Clean format
            ]
            
            # Remove None values and duplicates
            search_variations = list(dict.fromkeys([v for v in search_variations if v]))
            
            for variation in search_variations:
                patent = session.query(Patent).filter(
                    Patent.publication_number == variation
                ).first()
                if patent:
                    break
            
            if not patent:
                # If not found in database, fall back to semantic search
                return _handle_semantic_fallback_search(patent_number, original_query, context)
            
            # Format SQL database response
            response_parts = []
            response_parts.append(f"## Patent {patent.publication_number}")
            
            if patent.title_en:
                response_parts.append(f"**Title:** {patent.title_en}")
            
            if patent.abstract_text:
                # Truncate abstract if too long
                abstract_text = patent.abstract_text[:600] + "..." if len(patent.abstract_text) > 600 else patent.abstract_text
                response_parts.append(f"**Abstract:** {abstract_text}")
            
            if patent.publication_date:
                response_parts.append(f"**Publication Date:** {patent.publication_date}")
            
            if patent.publication_country:
                response_parts.append(f"**Country:** {patent.publication_country}")
            
            if patent.ipc:
                # Handle IPC codes (could be JSON string or direct value)
                ipc_display = patent.ipc
                if isinstance(patent.ipc, str) and patent.ipc.startswith('['):
                    try:
                        ipc_list = json.loads(patent.ipc)
                        ipc_display = ', '.join(ipc_list) if isinstance(ipc_list, list) else patent.ipc
                    except:
                        pass
                response_parts.append(f"**IPC Classification:** {ipc_display}")
            
            if patent.inventor_names:
                response_parts.append(f"**Inventors:** {patent.inventor_names}")
            
            if patent.applicant_names:
                response_parts.append(f"**Applicants:** {patent.applicant_names}")
            
            # Add applicant information if available
            if hasattr(patent, 'applicant_organization') and patent.applicant_organization:
                response_parts.append(f"**Applicant:** {patent.applicant_organization}")
            
            # Add citation information if available
            if hasattr(patent, 'citations_count') and patent.citations_count:
                response_parts.append(f"**Citations:** {patent.citations_count}")
            
            response_parts.append("")
            
            # Add the key question for more details
            response_parts.append("---")
            response_parts.append("💡 **Would you like more detailed information about this patent?**")
            response_parts.append("I can provide:")
            response_parts.append("• Detailed technical descriptions and claims")
            response_parts.append("• Complete patent document content")
            response_parts.append("• Technical specifications and implementation details")
            response_parts.append("• Related technology context")
            response_parts.append("")
            response_parts.append("Just ask: *'Show me more details about this patent'* or *'Get detailed claims for this patent'*")
            
            return ApiResponse(message="\n".join(response_parts))
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in SQL-first patent search: {e}")
        return ApiResponse(
            message=f"I encountered an error while looking up patent {patent_number}.",
            error=str(e)
        )

def _handle_semantic_fallback_search(patent_number: str, original_query: str, context: ConversationContext) -> ApiResponse:
    """
    Fallback to semantic search when patent not found in SQL database.
    """
    try:
        # Try semantic search for the patent number
        results = retriever.search(patent_number, top_k_return=10)
        
        if not results:
            return ApiResponse(
                message=f"I couldn't find patent {patent_number} in my database. "
                       f"Please verify the patent number is correct. You can also search for patents "
                       f"by technology keywords or inventor names."
            )
        
        # Check if any results actually contain the patent number
        relevant_results = []
        for result in results:
            content = result.get('text', '') + ' ' + result.get('title', '')
            if patent_number.upper() in content.upper():
                relevant_results.append(result)
        
        if relevant_results:
            response_parts = []
            response_parts.append(f"## Found information about {patent_number}")
            response_parts.append("*Note: This patent was found in the document database but not in the structured database.*")
            response_parts.append("")
            
            for i, result in enumerate(relevant_results[:3], 1):  # Top 3 relevant results
                if result.get('title'):
                    response_parts.append(f"**Reference {i}:**")
                    response_parts.append(f"Title: {result['title']}")
                
                content = result.get('text', '')
                if content:
                    # Extract a relevant snippet around the patent number
                    content_upper = content.upper()
                    patent_pos = content_upper.find(patent_number.upper())
                    if patent_pos >= 0:
                        start = max(0, patent_pos - 200)
                        end = min(len(content), patent_pos + 400)
                        snippet = content[start:end]
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                        response_parts.append(f"Content: {snippet}")
                    else:
                        # If patent number not found, show first part of content
                        preview = content[:400] + "..." if len(content) > 400 else content
                        response_parts.append(f"Content: {preview}")
                
                response_parts.append("")
            
            response_parts.append("---")
            response_parts.append("💡 **Would you like more detailed information about this patent?**")
            response_parts.append("Just ask: *'Show me more details about this patent'* or *'Get detailed information'*")
            
            return ApiResponse(message="\n".join(response_parts))
        
        else:
            # No relevant results found
            return ApiResponse(
                message=f"I couldn't find specific information about patent {patent_number}. "
                       f"Please verify the patent number is correct. I can help you search for patents "
                       f"using technology keywords, inventor names, or company names instead."
            )
            
    except Exception as e:
        logger.error(f"Error in semantic fallback search: {e}")
        return ApiResponse(
            message=f"I encountered an error while searching for patent {patent_number}.",
            error=str(e)
        )

# API endpoints
@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        session['user_id'] = session.get('user_id', str(uuid.uuid4()))
        query = request.json.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        # Get conversation context
        history = conversation_history.get(session['user_id'], [])
        context = ConversationContext(
            **get_context_from_history(history)
        )
        
        # Classify query type
        classification = classifier.classify_query(query)
        
        # Check which type of query we're dealing with
        query_type = None
        if classifier.should_execute_patent_detail(classification):
            query_type = 'patent_detail'
        elif classifier.should_execute_patent_search(classification):
            query_type = 'patent_search'
        elif classifier.should_execute_statistics(classification):
            query_type = 'statistics'
        else:
            query_type = 'conversation'
        
        # Route to appropriate handler
        handler_map = {
            'patent_detail': lambda q, c: handle_patent_detail(q, classifier.get_patent_number(classification), c),
            'patent_search': handle_search,
            'statistics': handle_stats,
            'conversation': handle_conversation
        }
        
        handler = handler_map.get(query_type, handle_conversation)
        response = handler(query, context)
        
        # Update conversation history
        history.append({"role": "user", "content": query})
        if response.message:
            history.append({"role": "assistant", "content": response.message})
        conversation_history[session['user_id']] = history
        
        return jsonify({
            "message": response.message,
            "chart": response.chart,
            "error": response.error,
            "insight": getattr(response, 'insight', ''),
            "takeaway": getattr(response, 'takeaway', '')
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    try:
        session['user_id'] = str(uuid.uuid4())
        conversation_history[session['user_id']] = []
        return jsonify({"message": "Conversation reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
