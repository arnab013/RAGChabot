from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import sys
import re
import uuid
from datetime import timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

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
    """Handle patent search queries"""
    try:
        # Get search results
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
    """Handle patent detail lookup queries"""
    try:
        from database.config import get_db_session_simple
        from database.models import Patent
        
        # Normalize patent number - remove common prefixes and handle different formats
        normalized_number = patent_number.upper()
        
        # Remove common prefixes
        for prefix in ['EP', 'US', 'WO']:
            if normalized_number.startswith(prefix):
                normalized_number = normalized_number[len(prefix):]
                break
        
        # Remove leading zeros and non-numeric suffixes (like B1, A1, etc.)
        # Extract just the numeric part
        import re
        numeric_match = re.search(r'(\d+)', normalized_number)
        if numeric_match:
            numeric_part = numeric_match.group(1).lstrip('0') or '0'
        else:
            numeric_part = normalized_number
        
        # Query database for the patent
        session = get_db_session_simple()
        
        try:
            # Try exact match first
            patent = session.query(Patent).filter(
                Patent.publication_number == patent_number
            ).first()
            
            # If not found, try without prefix
            if not patent and normalized_number != patent_number:
                patent = session.query(Patent).filter(
                    Patent.publication_number == normalized_number
                ).first()
            
            # If still not found, try with just the numeric part
            if not patent and numeric_part != normalized_number:
                patent = session.query(Patent).filter(
                    Patent.publication_number == numeric_part
                ).first()
            
            # If still not found, try with original number without leading zeros
            if not patent:
                original_no_zeros = patent_number.lstrip('0')
                if original_no_zeros != patent_number:
                    patent = session.query(Patent).filter(
                        Patent.publication_number == original_no_zeros
                    ).first()
            
            if not patent:
                return ApiResponse(
                    message=f"I apologize, but I couldn't find patent {patent_number} in my database. "
                           f"Please check the patent number and try again. You can also search for patents "
                           f"by technology or keywords to find relevant patents.",
                    error="Patent not found"
                )
            
            # Format the patent details response
            response = _format_patent_details(patent, patent_number)
            return ApiResponse(message=response)
            
        finally:
            session.close()
            
    except Exception as e:
        return ApiResponse(
            message=f"I encountered an error while looking up patent {patent_number}. Please try again.",
            error=str(e)
        )

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

def _batch_results(results: List[Dict], max_tokens: int = 8000) -> List[List[Dict]]:
    """Batch results to fit within token limits"""
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
    """Generate response for a batch of results"""
    prompt = _build_search_prompt(query, batch, context)
    try:
        return chat([{"role": "user", "content": prompt}])
    except Exception:
        return _generate_fallback_response(batch)

def _generate_summary(query: str, analysis: str, result_count: int) -> str:
    """Generate summary for multiple batches of results"""
    prompt = f"""Based on the following analyses of {result_count} patents related to "{query}", provide a comprehensive summary:

{analysis}

Please provide a concise summary highlighting the key findings, main themes, and most relevant patents."""
    
    try:
        return chat([{"role": "user", "content": prompt}])
    except Exception:
        return f"Analysis of {result_count} Patents:"

def _build_search_prompt(query: str, results: List[Dict], context: ConversationContext) -> str:
    """Build prompt for search results"""
    context_parts = []
    if context.latest_non_query:
        context_parts.append(f"User context: {context.latest_non_query}")
    if context.query_contexts:
        context_parts.append(f"Previous related queries: {'; '.join(context.query_contexts)}")
    
    prompt = [
        "You are a patent research assistant. Analyze the following patent information and provide a comprehensive response to the user's query.",
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
            f"- Text/Description: {result.get('text', '')[:500]}{'...' if len(result.get('text', '')) > 500 else ''}"
        ])
        
        if result.get('abstract'):
            prompt.append(f"- Abstract: {result['abstract'][:300]}{'...' if len(result['abstract']) > 300 else ''}")
    
    prompt.extend([
        "",
        "IMPORTANT: Focus on relevant patents only. Integrate patent references naturally. No similarity scores in response."
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
