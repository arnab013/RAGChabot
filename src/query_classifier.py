"""
Intelligent query classification using LLM to determine query types and needed operations.
"""
import re
import json
import logging
from typing import Dict, List, Optional

try:
    from llm_clients import chat
except ImportError:
    from .llm_clients import chat

logger = logging.getLogger(__name__)

class QueryClassifier:
    """
    Uses LLM to intelligently classify user queries and determine required operations.
    """
    
    def __init__(self):
        self.classification_prompt = self._create_classification_prompt()
    
    def _create_classification_prompt(self) -> str:
        """Create the system prompt for query classification."""
        return """You are an intelligent query classifier for a patent search and analytics system.

Your task is to analyze user queries and determine what type(s) of operations are needed. The system supports four main query types:

1. **CONVERSATION**: General chat, greetings, questions about the system, help requests, or non-patent/non-statistical queries
2. **PATENT_SEARCH**: Requests to find specific patents, inventions, or patent information based on technology, content, criteria, OR when users provide just a patent number for semantic search
3. **PATENT_DETAIL**: Requests for detailed database information about a specific patent using patent numbers WITH explicit detail keywords (e.g., "show details of EP0001556B1", "tell me about patent US1234567")
4. **STATISTICS**: Requests for data analysis, charts, trends, counts, distributions, or statistical insights about the patent database

CRITICAL DISTINCTION FOR PATENT QUERIES:
- If user provides JUST a patent number (e.g., "EP0002650B1", "US1234567"): → PATENT_SEARCH (semantic search)
- If user asks for details/information about a patent number (e.g., "details about EP0002650B1", "tell me about US1234567"): → PATENT_DETAIL (database lookup)

IMPORTANT CLASSIFICATION RULES:
- A single user query can require MULTIPLE query types (e.g., "Find solar energy patents and show me the publication trends")
- Always classify based on the USER'S INTENT, not just keywords
- Consider conversation context when provided
- Be precise - don't over-classify simple requests
- PATENT_DETAIL requires EXPLICIT detail-requesting keywords, not just a patent number alone

For each query, return a JSON response with:
{
    "query_types": ["TYPE1", "TYPE2", ...],
    "reasoning": "Brief explanation of classification",
    "confidence": 0.95,
    "specific_requests": {
        "patent_search": "extracted search terms or patent number for semantic search",
        "patent_detail": "extracted patent number for detailed lookup or null",
        "statistics": "specific statistics requested or null",
        "conversation": "conversational intent or null"
    }
}

EXAMPLES:

User: "Hello, how are you?"
Response: {
    "query_types": ["CONVERSATION"],
    "reasoning": "Simple greeting with no patent or statistical intent",
    "confidence": 0.99,
    "specific_requests": {"patent_search": null, "patent_detail": null, "statistics": null, "conversation": "greeting"}
}

User: "Find patents about solar energy"
Response: {
    "query_types": ["PATENT_SEARCH"],
    "reasoning": "Direct request to find specific patents by technology",
    "confidence": 0.98,
    "specific_requests": {"patent_search": "solar energy", "patent_detail": null, "statistics": null, "conversation": null}
}

User: "EP0002650B1"
Response: {
    "query_types": ["PATENT_SEARCH"],
    "reasoning": "Single patent number provided for semantic search, no detail keywords present",
    "confidence": 0.95,
    "specific_requests": {"patent_search": "EP0002650B1", "patent_detail": null, "statistics": null, "conversation": null}
}

User: "Show me details of patent EP0001556B1"
Response: {
    "query_types": ["PATENT_DETAIL"],
    "reasoning": "Request for detailed information about a specific patent number with explicit detail keywords",
    "confidence": 0.99,
    "specific_requests": {"patent_search": null, "patent_detail": "EP0001556B1", "statistics": null, "conversation": null}
}

User: "Tell me about patent US1234567"
Response: {
    "query_types": ["PATENT_DETAIL"],
    "reasoning": "Request for information about a specific patent number with explicit detail keywords",
    "confidence": 0.98,
    "specific_requests": {"patent_search": null, "patent_detail": "US1234567", "statistics": null, "conversation": null}
}

User: "How many patents are in the database?"
Response: {
    "query_types": ["STATISTICS"],
    "reasoning": "Request for database count statistics",
    "confidence": 0.97,
    "specific_requests": {"patent_search": null, "patent_detail": null, "statistics": "total patent count", "conversation": null}
}

User: "Show me the distribution of patents by SDG"
Response: {
    "query_types": ["STATISTICS"],
    "reasoning": "Request for data visualization and statistical analysis of SDG distribution",
    "confidence": 0.98,
    "specific_requests": {"patent_search": null, "patent_detail": null, "statistics": "SDG distribution visualization", "conversation": null}
}

User: "Which countries have the most patents?"
Response: {
    "query_types": ["STATISTICS"],
    "reasoning": "Request for statistical analysis and ranking of countries by patent count",
    "confidence": 0.96,
    "specific_requests": {"patent_search": null, "patent_detail": null, "statistics": "top countries by patent count", "conversation": null}
}

User: "Find artificial intelligence patents and show me the publication trends by year"
Response: {
    "query_types": ["PATENT_SEARCH", "STATISTICS"],
    "reasoning": "Combination request: search for AI patents AND trend analysis",
    "confidence": 0.95,
    "specific_requests": {"patent_search": "artificial intelligence", "patent_detail": null, "statistics": "publication trends by year", "conversation": null}
}

Now classify the following query:"""

    def extract_patent_number(self, query: str) -> Optional[str]:
        """Extract patent number from query using regex patterns."""
        # Common patent number patterns
        patterns = [
            r'\b(EP\d{7}[A-Z]\d?)\b',  # European patents: EP0001556B1
            r'\b(US\d{7,8}[A-Z]?\d?)\b',  # US patents: US1234567A1
            r'\b(WO\d{4}/\d{6}[A-Z]?\d?)\b',  # WIPO patents: WO2023/123456A1
            r'\b([A-Z]{2}\d{7,8}[A-Z]?\d?)\b',  # General format: XX1234567A1
            r'\b(\d{7,8}[A-Z]?\d?)\b',  # Just numbers with optional letter/digit
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def is_patent_detail_query(self, query: str) -> bool:
        """Check if query is asking for details about a specific patent."""
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
        
        # Check if query contains both a detail indicator and a patent number
        has_detail_indicator = any(indicator in query_lower for indicator in detail_indicators)
        has_patent_number = self.extract_patent_number(query) is not None
        
        # Special handling for "more details" type requests that might refer to previous context
        has_more_details_request = any(phrase in query_lower for phrase in [
            "more details", "more information", "detailed claims", 
            "get detailed", "show detailed", "comprehensive information"
        ])
        
        # If it's a "more details" request without explicit patent number, 
        # it might be referring to a previous patent discussion
        if has_more_details_request and not has_patent_number:
            # This might be a follow-up detail request
            return True
        
        # Special case: if the query is ONLY a patent number (or mostly just a patent number),
        # it should be treated as a search query, not a detail query
        if has_patent_number:
            # Remove the patent number from the query and see what's left
            patent_number = self.extract_patent_number(query)
            query_without_patent = query_lower.replace(patent_number.lower(), "").strip()
            
            # If after removing the patent number, there's very little left (just spaces, punctuation),
            # then this is just a patent number search, not a detail request
            remaining_words = [word for word in query_without_patent.split() if len(word) > 2]
            
            # If there are fewer than 2 meaningful words left, treat as search
            if len(remaining_words) < 2:
                return False
        
        return has_detail_indicator and (has_patent_number or has_more_details_request)

    def classify_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> Dict:
        """
        Classify a user query to determine required operations.
        
        Args:
            user_query: The user's input query
            conversation_context: Previous conversation messages for context
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Build the full prompt
            prompt = self.classification_prompt
            
            # Add conversation context if available
            if conversation_context:
                context_str = "\n\nConversation Context (last few messages):\n"
                for msg in conversation_context[-3:]:  # Last 3 messages for context
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]  # Limit length
                    context_str += f"{role.title()}: {content}\n"
                prompt += context_str
            
            prompt += f'\n\nUser Query: "{user_query}"\n\nClassification (JSON only):'
            
            # Get LLM classification
            response = chat([{"role": "user", "content": prompt}])
            
            # Parse the JSON response
            try:
                # Extract JSON from response (in case there's extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    classification = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM classification response: {e}")
                # Fallback to keyword-based classification
                return self._fallback_classification(user_query)
            
            # Validate and normalize the classification
            return self._validate_classification(classification, user_query)
            
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            # Fallback to keyword-based classification
            return self._fallback_classification(user_query)
    
    def _validate_classification(self, classification: Dict, user_query: str) -> Dict:
        """Validate and normalize the LLM classification response."""
        try:
            # Ensure required fields exist
            if "query_types" not in classification:
                raise ValueError("Missing query_types field")
              # Normalize query types
            valid_types = {"CONVERSATION", "PATENT_SEARCH", "PATENT_DETAIL", "STATISTICS"}
            query_types = [qt.upper().replace(" ", "_") for qt in classification["query_types"]]
            query_types = [qt for qt in query_types if qt in valid_types]
            
            if not query_types:
                query_types = ["CONVERSATION"]  # Default fallback
            
            # Double-check for patent detail queries using regex
            if self.is_patent_detail_query(user_query) and "PATENT_DETAIL" not in query_types:
                query_types.append("PATENT_DETAIL")
            
            # Ensure confidence is reasonable
            confidence = float(classification.get("confidence", 0.7))
            confidence = max(0.1, min(1.0, confidence))
            
            # Clean up specific requests
            specific_requests = classification.get("specific_requests", {})
            if not isinstance(specific_requests, dict):
                specific_requests = {"patent_search": None, "patent_detail": None, "statistics": None, "conversation": None}
            
            # Extract patent number if it's a detail query
            if "PATENT_DETAIL" in query_types and not specific_requests.get("patent_detail"):
                specific_requests["patent_detail"] = self.extract_patent_number(user_query)
            
            return {
                "query_types": query_types,
                "reasoning": classification.get("reasoning", "LLM classification"),
                "confidence": confidence,
                "specific_requests": specific_requests,
                "original_query": user_query
            }
            
        except Exception as e:
            logger.warning(f"Classification validation failed: {e}")
            return self._fallback_classification(user_query)
    
    def _fallback_classification(self, user_query: str) -> Dict:
        """Fallback keyword-based classification when LLM fails."""
        query_lower = user_query.lower()
        
        # Patent keywords
        patent_keywords = [
            "patent", "invention", "uspto", "prior art", "cite", "citation",
            "application", "granted", "filing date", "priority date", "assignee",
            "inventor", "IPC", "USPC", "CPC", "classification", "wipo", "novelty",
            "innovation", "intellectual property", "find patents", "search patents"
        ]
          # Statistics keywords
        stats_keywords = [
            "how many", "total", "count", "statistics", "breakdown", "distribution",
            "trends", "by year", "by country", "chart", "graph", "visualize", "show me",
            "top companies", "top inventors", "sdg distribution", "technology fields",
            "plot", "display", "comparison", "compare", "analyze", "overview",
            "what countries", "which countries", "main technology", "technology breakdown",
            "patent trends", "publication trends", "sdg", "goal", "field", "applicant"        ]
        
        # Conversational keywords
        conversation_keywords = [
            "hello", "hi", "help", "how are you", "what can you do", "thanks",
            "goodbye", "explain", "tell me about"
        ]
        
        query_types = []
        specific_requests = {"patent_search": None, "patent_detail": None, "statistics": None, "conversation": None}
        
        # Check for patent detail queries first (most specific)
        if self.is_patent_detail_query(user_query):
            query_types.append("PATENT_DETAIL")
            specific_requests["patent_detail"] = self.extract_patent_number(user_query)
        
        # Check for standalone patent number (semantic search)
        elif self.extract_patent_number(user_query) and not self.is_patent_detail_query(user_query):
            # If we found a patent number but it's not a detail query, treat as search
            query_types.append("PATENT_SEARCH")
            specific_requests["patent_search"] = user_query
        
        # Check for patent search with keywords
        elif any(keyword in query_lower for keyword in patent_keywords):
            query_types.append("PATENT_SEARCH")
            specific_requests["patent_search"] = user_query
        
        # Check for statistics
        if any(keyword in query_lower for keyword in stats_keywords):
            query_types.append("STATISTICS")
            specific_requests["statistics"] = user_query
        
        # Check for conversation
        if any(keyword in query_lower for keyword in conversation_keywords) or not query_types:
            query_types.append("CONVERSATION")
            specific_requests["conversation"] = user_query
        
        # Check for patent detail
        if self.is_patent_detail_query(user_query):
            query_types.append("PATENT_DETAIL")
            specific_requests["patent_detail"] = self.extract_patent_number(user_query)
        
        # Default to conversation if nothing else matches
        if not query_types:
            query_types = ["CONVERSATION"]
            specific_requests["conversation"] = user_query
        
        return {
            "query_types": query_types,
            "reasoning": "Fallback keyword-based classification",
            "confidence": 0.6,
            "specific_requests": specific_requests,
            "original_query": user_query
        }
    
    def should_execute_patent_search(self, classification: Dict) -> bool:
        """Determine if a patent search should be executed."""
        return "PATENT_SEARCH" in classification.get("query_types", [])
    
    def should_execute_statistics(self, classification: Dict) -> bool:
        """Determine if statistical analysis should be executed."""
        return "STATISTICS" in classification.get("query_types", [])
    
    def should_execute_conversation(self, classification: Dict) -> bool:
        """Determine if conversational response should be generated."""
        return "CONVERSATION" in classification.get("query_types", [])
    
    def should_execute_patent_detail(self, classification: Dict) -> bool:
        """Determine if a patent detail lookup should be executed."""
        return "PATENT_DETAIL" in classification.get("query_types", [])
    
    def get_search_terms(self, classification: Dict) -> Optional[str]:
        """Extract search terms for patent search."""
        specific = classification.get("specific_requests", {})
        return specific.get("patent_search")
    
    def get_statistics_request(self, classification: Dict) -> Optional[str]:
        """Extract statistics request details."""
        specific = classification.get("specific_requests", {})
        return specific.get("statistics")
    
    def get_conversation_intent(self, classification: Dict) -> Optional[str]:
        """Extract conversational intent."""
        specific = classification.get("specific_requests", {})
        return specific.get("conversation")
    
    def get_patent_number(self, classification: Dict) -> Optional[str]:
        """Extract patent number for patent detail lookup."""
        specific = classification.get("specific_requests", {})
        return specific.get("patent_detail")
