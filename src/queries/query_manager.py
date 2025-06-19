"""
Query Manager - Coordinates all specialized query handlers
"""
from typing import Dict, List, Any, Optional
from .base import BaseQueryHandler, QueryResponse

# Import handlers with error handling
try:
    from .publication_trends import PublicationTrendsHandler
    from .sdg_distribution import SDGDistributionHandler
    from .technology_analysis import TechnologyAnalysisHandler
    from .inventor_assignee import InventorAssigneeHandler
    from .geographical_analysis import GeographicalAnalysisHandler
except ImportError as e:
    import logging
    logging.error(f"Error importing handlers: {e}")
    raise


class QueryManager:
    """Manages and routes queries to appropriate handlers"""
    
    def __init__(self):
        # Initialize all handlers
        self.handlers = {
            'publication_trends': PublicationTrendsHandler(),
            'sdg_distribution': SDGDistributionHandler(),
            'technology_analysis': TechnologyAnalysisHandler(),
            'inventor_assignee': InventorAssigneeHandler(),
            'geographical_analysis': GeographicalAnalysisHandler()
        }
          # Build keyword mapping for quick lookup
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self) -> Dict[str, str]:
        """Build mapping from keywords to handler names"""
        keyword_map = {}
        
        for handler_name, handler in self.handlers.items():
            keywords = handler.get_query_keywords()
            for keyword in keywords:
                keyword_map[keyword.lower()] = handler_name
        
        return keyword_map
    
    def route_query(self, query: str, **kwargs) -> QueryResponse:
        """Route query to appropriate handler"""
        try:
            # Determine the best handler for this query
            handler_name = self._identify_handler(query)
            
            if handler_name and handler_name in self.handlers:
                handler = self.handlers[handler_name]
                try:
                    result = handler.handle_query(query, **kwargs)
                    # Check if the result has an error message indicating failure
                    if result.message and "Error:" in result.message:
                        print(f"SQL handler {handler_name} failed, falling back to semantic search...")
                        return self._fallback_to_semantic_search(query, **kwargs)
                    return result
                except Exception as e:
                    print(f"SQL handler {handler_name} threw exception: {str(e)}, falling back to semantic search...")
                    return self._fallback_to_semantic_search(query, **kwargs)
            else:
                # No specific handler found, use semantic search
                return self._fallback_to_semantic_search(query, **kwargs)
        except Exception as e:
            # Generate dynamic error message for general query manager failures
            try:
                from ..llm_clients import chat
                
                prompt = f"""
A user asked: "{query}"

The patent analytics system encountered an unexpected error while processing this query.

Generate a helpful, user-friendly message that:
1. Acknowledges their request
2. Explains that there was a system issue
3. Suggests they try rephrasing their question or try a simpler query
4. Maintains a professional and helpful tone

Keep it concise (2-3 sentences) and avoid technical details.
"""
                
                messages = [{"role": "user", "content": prompt}]
                error_message = chat(messages, temperature=0.7, max_tokens=150)
                return QueryResponse(message=error_message.strip())
                
            except Exception:
                # Only use placeholder when LLM is unavailable
                return QueryResponse(
                    message="My server is currently under maintenance. Please try again later or contact the developer for assistance."
                )
    
    def _fallback_to_semantic_search(self, query: str, **kwargs) -> QueryResponse:
        """Fallback to semantic search when SQL handlers fail"""
        try:
            # Import here to avoid circular imports
            from ..retrieval import PatentRetriever
            from ..summarise import PatentSummarizer
            
            # Use semantic search
            retriever = PatentRetriever()
            chunks = retriever.search(query, k=10)
            
            if chunks:
                # Use summarizer to generate response
                summarizer = PatentSummarizer()
                response = summarizer.generate_summary(chunks, query)
                
                return QueryResponse(
                    message=response,
                    data={'search_results': [{'content': chunk.content, 'metadata': chunk.metadata} for chunk in chunks]}
                )
            else:
                return QueryResponse(
                    message="I couldn't find relevant information for your query. Please try rephrasing or asking about specific aspects of the patent data."
                )
                
        except Exception as e:
            return QueryResponse(
                message=f"I apologize, but I'm having trouble processing your query right now. Please try again later."
            )
    
    def _identify_handler(self, query: str) -> Optional[str]:
        """Identify which handler should process the query"""
        query_lower = query.lower()
        
        # First check for exact multi-word matches (prioritize more specific phrases)
        for keyword, handler_name in self.keyword_map.items():
            if ' ' in keyword and keyword in query_lower:  # Multi-word keywords have priority
                return handler_name
        
        # Count keyword matches for each handler
        handler_scores = {}
        
        for keyword, handler_name in self.keyword_map.items():
            if keyword in query_lower:
                if handler_name not in handler_scores:
                    handler_scores[handler_name] = 0
                # Give higher weight to longer, more specific keywords
                weight = len(keyword.split()) * 2 if ' ' in keyword else 1
                handler_scores[handler_name] += weight
        
        # Return handler with highest score
        if handler_scores:
            return max(handler_scores.items(), key=lambda x: x[1])[0]
        
        # Default handler if no specific match
        return 'publication_trends'
    
    def get_available_queries(self) -> Dict[str, List[str]]:
        """Get list of available query types and their keywords"""
        return {
            handler_name: handler.get_query_keywords()
            for handler_name, handler in self.handlers.items()
        }
    
    def close_all(self):
        """Close all handlers"""
        for handler in self.handlers.values():
            try:
                handler.close()
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_tb, exc_traceback):
        self.close_all()


# Convenience function for direct use
def handle_query(query: str, **kwargs) -> Dict[str, Any]:
    """Handle a query using the appropriate handler"""
    with QueryManager() as manager:
        response = manager.route_query(query, **kwargs)
        return response.to_dict()
