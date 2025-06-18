"""
Chart generation engine for patent data visualization.
Implements the enhanced chart generation workflow.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.config import get_db_session_simple
from .chart_templates import chart_registry, ChartTemplate, ChartType

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Main chart generation engine"""
    
    def __init__(self):
        self.registry = chart_registry
    
    def generate_chart(self, template_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate chart data using a template
        
        Args:
            template_id: ID of the chart template to use
            parameters: Template parameters (limit, sdg_number, date ranges, etc.)
            
        Returns:
            Dict containing chart data and metadata
        """
        if parameters is None:
            parameters = {}
            
        template = self.registry.get_template(template_id)
        if not template:
            raise ValueError(f"Unknown template ID: {template_id}")
        
        try:
            # Merge template default parameters with provided parameters
            merged_params = {**template.parameters, **parameters}
            
            # Build SQL filters
            filters = self.registry.build_sql_filters(merged_params)
            
            # Format SQL query with filters
            sql_query = template.sql_query.format(**filters)
            
            # Execute query
            raw_data = self._execute_query(sql_query)
            
            # Format data for frontend
            chart_data = self._format_chart_data(raw_data, template, merged_params)
            
            return {
                'type': template.chart_type.value,
                'title': template.description,
                'data': chart_data,
                'template_id': template_id,
                'parameters': merged_params,
                'sql_query': sql_query,  # For debugging
                'data_source': template.data_source,
                'notes': template.notes
            }
            
        except Exception as e:
            logger.error(f"Error generating chart for template {template_id}: {str(e)}")
            logger.error(f"SQL Query: {sql_query if 'sql_query' in locals() else 'Not generated'}")
            raise    
    def _execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        session = get_db_session_simple()
        try:
            result = session.execute(text(sql_query))
            columns = result.keys()
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                data.append(row_dict)
            
            logger.info(f"Query executed successfully, returned {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            logger.error(f"Query: {sql_query}")
            raise
        finally:
            session.close()
    
    def _format_chart_data(self, raw_data: List[Dict[str, Any]], template: ChartTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw query results for frontend chart consumption"""
        
        if not raw_data:
            return {'labels': [], 'datasets': []}
        
        chart_type = template.chart_type
        
        # Color palettes
        colors = [
            '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
            '#06B6D4', '#F97316', '#84CC16', '#EC4899', '#6366F1',
            '#14B8A6', '#F472B6', '#A855F7', '#22D3EE', '#FB7185'
        ]
        
        if chart_type == ChartType.PIE or chart_type == ChartType.DONUT:
            return self._format_pie_data(raw_data, colors)
        
        elif chart_type in [ChartType.LINE, ChartType.AREA]:
            return self._format_line_area_data(raw_data, colors, chart_type == ChartType.AREA)
        
        elif chart_type == ChartType.BAR:
            return self._format_bar_data(raw_data, colors)
        
        elif chart_type == ChartType.TREEMAP:
            return self._format_treemap_data(raw_data, colors)
        
        elif chart_type == ChartType.GROUPED_BAR:
            return self._format_grouped_bar_data(raw_data, colors)
        
        elif chart_type == ChartType.HISTOGRAM:
            return self._format_histogram_data(raw_data, colors)
        
        else:
            # Default to bar chart format
            return self._format_bar_data(raw_data, colors)
    
    def _format_pie_data(self, raw_data: List[Dict], colors: List[str]) -> Dict[str, Any]:
        """Format data for pie/donut charts"""
        return {
            'labels': [str(item.get('category', item.get('name', 'Unknown'))) for item in raw_data],
            'datasets': [{
                'data': [item.get('count', item.get('value', 0)) for item in raw_data],
                'backgroundColor': colors[:len(raw_data)],
                'borderWidth': 2,
                'borderColor': '#ffffff'
            }]
        }
    
    def _format_line_area_data(self, raw_data: List[Dict], colors: List[str], is_area: bool = False) -> Dict[str, Any]:
        """Format data for line/area charts"""
        return {
            'labels': [str(item.get('category', item.get('year', item.get('month', 'Unknown')))) for item in raw_data],
            'datasets': [{
                'label': 'Patents',
                'data': [item.get('count', item.get('value', 0)) for item in raw_data],
                'borderColor': colors[0],
                'backgroundColor': f"{colors[0]}30" if is_area else f"{colors[0]}10",
                'fill': is_area,
                'tension': 0.4,
                'pointBackgroundColor': colors[0],
                'pointBorderColor': '#ffffff',
                'pointBorderWidth': 2,
                'pointRadius': 4
            }]
        }
    
    def _format_bar_data(self, raw_data: List[Dict], colors: List[str]) -> Dict[str, Any]:
        """Format data for bar charts"""
        return {
            'labels': [str(item.get('category', item.get('name', 'Unknown'))) for item in raw_data],
            'datasets': [{
                'label': 'Patents',
                'data': [item.get('count', item.get('value', 0)) for item in raw_data],
                'backgroundColor': colors[0],
                'borderColor': colors[0],
                'borderWidth': 1,
                'borderRadius': 4,
                'borderSkipped': False
            }]
        }
    
    def _format_treemap_data(self, raw_data: List[Dict], colors: List[str]) -> List[Dict[str, Any]]:
        """Format data for treemap charts"""
        return [
            {
                'name': str(item.get('category', item.get('name', 'Unknown'))),
                'value': item.get('count', item.get('value', 0)),
                'fill': colors[i % len(colors)]
            }
            for i, item in enumerate(raw_data)
        ]
    
    def _format_grouped_bar_data(self, raw_data: List[Dict], colors: List[str]) -> Dict[str, Any]:
        """Format data for grouped bar charts"""
        # Extract all possible data series keys (excluding 'category')
        if not raw_data:
            return {'labels': [], 'datasets': []}
        
        first_item = raw_data[0]
        value_keys = [k for k in first_item.keys() if k != 'category' and isinstance(first_item[k], (int, float))]
        
        datasets = []
        for i, key in enumerate(value_keys):
            datasets.append({
                'label': key.replace('_', ' ').title(),
                'data': [item.get(key, 0) for item in raw_data],
                'backgroundColor': colors[i % len(colors)],
                'borderColor': colors[i % len(colors)],
                'borderWidth': 1,
                'borderRadius': 4
            })
        
        return {
            'labels': [str(item.get('category', 'Unknown')) for item in raw_data],
            'datasets': datasets
        }
    
    def _format_histogram_data(self, raw_data: List[Dict], colors: List[str]) -> Dict[str, Any]:
        """Format data for histogram charts"""
        return {
            'labels': [str(item.get('category', item.get('bucket', 'Unknown'))) for item in raw_data],
            'datasets': [{
                'label': 'Frequency',
                'data': [item.get('count', item.get('value', 0)) for item in raw_data],
                'backgroundColor': colors[2],  # Different color for histograms
                'borderColor': colors[2],
                'borderWidth': 1,
                'borderRadius': 2
            }]
        }
    
    def suggest_templates(self, query_context: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Suggest appropriate chart templates based on query context
        
        Args:
            query_context: User's query or context
            limit: Maximum number of suggestions
            
        Returns:
            List of template suggestions with relevance scores
        """
        query_lower = query_context.lower()
        
        # Keywords to template mapping
        keyword_mapping = {
            'yearly_count': ['year', 'annual', 'time', 'trend', 'timeline', 'over time'],
            'sdg_distribution': ['sdg', 'goal', 'sustainable', 'distribution'],
            'kind_breakdown': ['kind', 'type', 'publication', 'breakdown'],
            'geo_distribution': ['country', 'geographic', 'location', 'where', 'applicant'],
            'ipc_treemap': ['technology', 'ipc', 'field', 'classification', 'tech'],
            'family_sizes': ['family', 'size', 'related', 'group'],
            'chunk_counts': ['chunk', 'text', 'segment'],
            'monthly_timeline': ['month', 'monthly', 'recent', 'timeline'],
            'app_vs_inv_countries': ['applicant', 'inventor', 'comparison', 'vs', 'versus']
        }
        
        # Calculate relevance scores
        suggestions = []
        for template_id, keywords in keyword_mapping.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                template = self.registry.get_template(template_id)
                suggestions.append({
                    'template_id': template_id,
                    'description': template.description,
                    'relevance_score': score,
                    'chart_type': template.chart_type.value
                })
        
        # Sort by relevance and return top suggestions
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return suggestions[:limit]
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get all available templates for LLM context"""
        return self.registry.get_template_descriptions()

# Global chart generator instance
chart_generator = ChartGenerator()
