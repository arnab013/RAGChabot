"""
Query Manager - Coordinates all specialized query handlers
"""
from typing import Dict, List, Any, Optional
from .base import BaseQueryHandler, QueryResponse
from .publication_trends import PublicationTrendsHandler
from .sdg_distribution import SDGDistributionHandler
from .technology_analysis import TechnologyAnalysisHandler
from .inventor_assignee import InventorAssigneeHandler
from .geographical_analysis import GeographicalAnalysisHandler


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
                return handler.handle_query(query, **kwargs)
            else:
                # Default fallback
                return QueryResponse(
                    message="I'm not sure how to handle that query. Please try asking about publication trends, SDG distribution, technology analysis, inventors, assignees, or geographical distribution."
                )
                
        except Exception as e:
            return QueryResponse(
                message=f"Sorry, I encountered an error processing your query: {str(e)}"
            )
    
    def _identify_handler(self, query: str) -> Optional[str]:
        """Identify which handler should process the query"""
        query_lower = query.lower()
        
        # Count keyword matches for each handler
        handler_scores = {}
        
        for keyword, handler_name in self.keyword_map.items():
            if keyword in query_lower:
                if handler_name not in handler_scores:
                    handler_scores[handler_name] = 0
                handler_scores[handler_name] += 1
        
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
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


# Convenience function for direct use
def handle_query(query: str, **kwargs) -> Dict[str, Any]:
    """Handle a query using the appropriate handler"""
    with QueryManager() as manager:
        response = manager.route_query(query, **kwargs)
        return response.to_dict()


# Test function
def test_query_manager():
    """Test the query manager with various queries"""
    test_queries = [
        "Show me patent publication trends in last 12 months",
        "What's the SDG distribution?", 
        "Technology analysis by CPC classification",
        "Top 10 inventors",
        "Leading assignee companies",
        "Geographical distribution by country",
        "Compare publication trends 2023 vs 2025",
        "Trends in 2024"
    ]
    
    with QueryManager() as manager:
        print("=== Testing Query Manager ===\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            handler_name = manager._identify_handler(query)
            print(f"Handler: {handler_name}")
            
            try:
                response = manager.route_query(query)
                print(f"Response: {response.message[:100]}...")
                print(f"Chart: {'Yes' if response.chart else 'No'}")
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 50)


if __name__ == "__main__":
    test_query_manager()
