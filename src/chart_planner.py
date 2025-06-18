"""
LLM integration for intelligent chart planning and generation.
Handles chart type selection and parameter extraction from user queries.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .chart_engine import chart_generator
from .chart_templates import ChartType

logger = logging.getLogger(__name__)

@dataclass
class ChartPlan:
    """Represents a planned chart with template and parameters"""
    template_id: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

class ChartPlanner:
    """Intelligent chart planning using LLM-style analysis"""
    
    def __init__(self):
        self.chart_gen = chart_generator
        
    def plan_chart(self, user_query: str, context: str = "") -> Optional[ChartPlan]:
        """
        Analyze user query and plan appropriate chart generation
        
        Args:
            user_query: User's question or request
            context: Additional context from conversation
            
        Returns:
            ChartPlan with template selection and parameters
        """
        try:
            # Extract intent and parameters from query
            intent_analysis = self._analyze_query_intent(user_query)
            
            if not intent_analysis['is_chart_request']:
                return None
            
            # Select best template
            template_selection = self._select_template(user_query, intent_analysis)
            
            if not template_selection:
                return None
            
            # Extract parameters
            parameters = self._extract_parameters(user_query, template_selection['template_id'])
            
            return ChartPlan(
                template_id=template_selection['template_id'],
                parameters=parameters,
                confidence=template_selection['confidence'],
                reasoning=template_selection['reasoning']
            )
            
        except Exception as e:
            logger.error(f"Error in chart planning: {str(e)}")
            return None
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze if query requests chart/visualization"""
        query_lower = query.lower()
        
        # Chart request indicators
        chart_keywords = [
            'show', 'chart', 'graph', 'plot', 'visualize', 'display',
            'distribution', 'breakdown', 'trend', 'over time',
            'comparison', 'compare', 'analyze', 'statistics'
        ]
        
        visualization_patterns = [
            r'show me.*distribution',
            r'how many.*per',
            r'what.*trend',
            r'compare.*between',
            r'breakdown.*by',
            r'visualize.*',
            r'chart.*showing'
        ]
        
        # Check for explicit chart requests
        has_chart_keyword = any(keyword in query_lower for keyword in chart_keywords)
        has_visualization_pattern = any(re.search(pattern, query_lower) for pattern in visualization_patterns)
        
        # Numerical/statistical question indicators
        numerical_patterns = [
            r'how many',
            r'what.*number',
            r'count.*',
            r'percentage.*',
            r'proportion.*'
        ]
        
        has_numerical_intent = any(re.search(pattern, query_lower) for pattern in numerical_patterns)
        
        is_chart_request = has_chart_keyword or has_visualization_pattern or has_numerical_intent
        
        return {
            'is_chart_request': is_chart_request,
            'has_explicit_chart_request': has_chart_keyword or has_visualization_pattern,
            'has_numerical_intent': has_numerical_intent,
            'confidence': 0.9 if has_chart_keyword else 0.7 if has_visualization_pattern else 0.5
        }
    
    def _select_template(self, query: str, intent_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the most appropriate chart template"""
        query_lower = query.lower()
        
        # Template selection rules with confidence scores
        template_rules = [
            # Time-based queries
            {
                'template_id': 'yearly_count',
                'keywords': ['year', 'annual', 'over time', 'timeline', 'trend', 'time'],
                'patterns': [r'per year', r'over.*years?', r'annual', r'yearly'],
                'confidence_boost': 0.2
            },
            {
                'template_id': 'monthly_timeline',
                'keywords': ['month', 'monthly', 'recent'],
                'patterns': [r'per month', r'monthly', r'last.*months?'],
                'confidence_boost': 0.2
            },
            
            # SDG-related queries
            {
                'template_id': 'sdg_distribution',
                'keywords': ['sdg', 'goal', 'sustainable development', 'distribution'],
                'patterns': [r'sdg.*distribution', r'sustainable.*goal', r'goal.*\d+'],
                'confidence_boost': 0.3
            },
            
            # Geographic queries
            {
                'template_id': 'geo_distribution',
                'keywords': ['country', 'countries', 'geographic', 'location', 'where', 'applicant'],
                'patterns': [r'by country', r'countries.*most', r'geographic.*distribution'],
                'confidence_boost': 0.2
            },
            
            # Technology/IPC queries
            {
                'template_id': 'ipc_treemap',
                'keywords': ['technology', 'tech', 'ipc', 'field', 'classification'],
                'patterns': [r'technology.*field', r'tech.*area', r'ipc.*class'],
                'confidence_boost': 0.2
            },
            
            # Publication type queries
            {
                'template_id': 'kind_breakdown',
                'keywords': ['kind', 'type', 'publication type', 'breakdown'],
                'patterns': [r'publication.*type', r'kind.*patent', r'type.*distribution'],
                'confidence_boost': 0.2
            },
            
            # Family/relationship queries
            {
                'template_id': 'family_sizes',
                'keywords': ['family', 'related', 'size', 'group'],
                'patterns': [r'family.*size', r'related.*patent', r'patent.*group'],
                'confidence_boost': 0.2
            },
            
            # Comparison queries
            {
                'template_id': 'app_vs_inv_countries',
                'keywords': ['applicant', 'inventor', 'comparison', 'compare', 'vs', 'versus'],
                'patterns': [r'applicant.*inventor', r'inventor.*applicant', r'compare.*country'],
                'confidence_boost': 0.3
            }
        ]
        
        # Score each template
        template_scores = []
        
        for rule in template_rules:
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in rule['keywords'] if keyword in query_lower)
            score += keyword_matches * 0.3
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in rule['patterns'] if re.search(pattern, query_lower))
            score += pattern_matches * 0.4
            
            # Apply confidence boost
            if score > 0:
                score += rule['confidence_boost']
            
            if score > 0:
                template_scores.append({
                    'template_id': rule['template_id'],
                    'score': score,
                    'keyword_matches': keyword_matches,
                    'pattern_matches': pattern_matches
                })
        
        # If no specific template matches well, use suggestions
        if not template_scores or max(s['score'] for s in template_scores) < 0.5:
            suggestions = self.chart_gen.suggest_templates(query_lower, limit=3)
            if suggestions:
                best_suggestion = suggestions[0]
                return {
                    'template_id': best_suggestion['template_id'],
                    'confidence': 0.6,
                    'reasoning': f"Suggested based on keywords: {best_suggestion['description']}"
                }
        
        # Return best scoring template
        if template_scores:
            best_template = max(template_scores, key=lambda x: x['score'])
            template = self.chart_gen.registry.get_template(best_template['template_id'])
            
            return {
                'template_id': best_template['template_id'],
                'confidence': min(best_template['score'], 0.95),
                'reasoning': f"Selected {template.description} based on {best_template['keyword_matches']} keywords and {best_template['pattern_matches']} patterns"
            }
        
        return None
    
    def _extract_parameters(self, query: str, template_id: str) -> Dict[str, Any]:
        """Extract parameters from query for the selected template"""
        parameters = {}
        query_lower = query.lower()
        
        # Extract limit/count parameters
        limit_patterns = [
            r'top (\d+)',
            r'first (\d+)',
            r'(\d+) most',
            r'limit (\d+)',
            r'show (\d+)'
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters['limit'] = int(match.group(1))
                break
        
        # Extract year parameters
        year_patterns = [
            r'in (\d{4})',
            r'from (\d{4})',
            r'since (\d{4})',
            r'after (\d{4})',
            r'(\d{4})\s*to\s*(\d{4})',
            r'between\s*(\d{4})\s*and\s*(\d{4})'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:  # Range pattern
                    parameters['start_year'] = int(match.group(1))
                    parameters['end_year'] = int(match.group(2))
                else:  # Single year pattern
                    year = int(match.group(1))
                    # Determine if it's start or end year based on context
                    if any(word in query_lower for word in ['from', 'since', 'after']):
                        parameters['start_year'] = year
                    elif any(word in query_lower for word in ['before', 'until', 'to']):
                        parameters['end_year'] = year
                    else:
                        parameters['start_year'] = year
                        parameters['end_year'] = year
                break
        
        # Extract SDG number
        sdg_patterns = [
            r'sdg (\d+)',
            r'goal (\d+)',
            r'sustainable development goal (\d+)'
        ]
        
        for pattern in sdg_patterns:
            match = re.search(pattern, query_lower)
            if match:
                sdg_num = int(match.group(1))
                if 1 <= sdg_num <= 17:
                    parameters['sdg_number'] = sdg_num
                break
        
        # Template-specific default limits
        default_limits = {
            'yearly_count': 10,
            'sdg_distribution': 17,  # All SDGs
            'geo_distribution': 10,
            'ipc_treemap': 20,
            'kind_breakdown': 8,
            'family_sizes': 5,
            'chunk_counts': 5,
            'monthly_timeline': 24,
            'app_vs_inv_countries': 10
        }
        
        if 'limit' not in parameters and template_id in default_limits:
            parameters['limit'] = default_limits[template_id]
        
        return parameters
    
    def generate_chart_response(self, chart_plan: ChartPlan) -> Dict[str, Any]:
        """Generate chart using the planned template and parameters"""
        try:
            chart_result = self.chart_gen.generate_chart(
                chart_plan.template_id,
                chart_plan.parameters
            )
            
            # Add planning metadata
            chart_result['plan'] = {
                'confidence': chart_plan.confidence,
                'reasoning': chart_plan.reasoning,
                'template_id': chart_plan.template_id,
                'parameters': chart_plan.parameters
            }
            
            return chart_result
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            raise

# Global chart planner instance
chart_planner = ChartPlanner()
