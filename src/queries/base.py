"""
Base classes and utilities for query handlers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from database.config import get_db_session_simple
from database.models import Patent
import logging

# Import LLM for dynamic insight generation
try:
    from llm_clients import chat
except ImportError:
    from .llm_clients import chat

logger = logging.getLogger(__name__)


class BaseQueryHandler(ABC):
    """Base class for all query handlers"""
    
    def __init__(self):
        self.session = get_db_session_simple()
    
    @abstractmethod
    def handle_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle the specific query type"""
        pass
    
    @abstractmethod
    def get_query_keywords(self) -> List[str]:
        """Return keywords that identify this query type"""
        pass
    
    def generate_dynamic_insights(self, query: str, chart_data: Dict, data_summary: str) -> Dict[str, str]:
        """Generate dynamic insights and takeaways using LLM based on chart data"""
        try:
            # Prepare chart data summary for LLM
            chart_type = chart_data.get('type', 'unknown')
            chart_title = chart_data.get('title', 'Data Analysis')
            
            # Extract key data points
            data_points = []
            if 'data' in chart_data and 'labels' in chart_data['data'] and 'datasets' in chart_data['data']:
                labels = chart_data['data']['labels']
                datasets = chart_data['data']['datasets']
                
                if datasets and len(datasets) > 0:
                    values = datasets[0].get('data', [])
                    for i, label in enumerate(labels[:min(len(labels), len(values))]):
                        if i < len(values):
                            data_points.append(f"{label}: {values[i]}")
            
            # Construct LLM prompt
            prompt = f"""Based on the following patent data analysis, generate concise and insightful insights and takeaways:

Query: {query}
Chart Type: {chart_type}
Chart Title: {chart_title}
Data Summary: {data_summary}
Key Data Points: {', '.join(data_points[:8])}

Please provide:
1. A brief insight (1-2 sentences) highlighting the most important trend or pattern in the data
2. A practical takeaway (1-2 sentences) explaining what this means for innovation strategy or business decisions

Format your response as JSON:
{{
    "insight": "Your insight here",
    "takeaway": "Your takeaway here"
}}"""

            messages = [{"role": "user", "content": prompt}]
            
            # Call LLM
            response = chat(messages, temperature=0.3, max_tokens=200)
            
            # Parse LLM response
            import json
            try:
                parsed_response = json.loads(response)
                return {
                    "insight": parsed_response.get("insight", "This chart shows interesting patterns in the patent data."),
                    "takeaway": parsed_response.get("takeaway", "These insights can inform strategic innovation decisions.")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                lines = response.strip().split('\n')
                insight = ""
                takeaway = ""
                
                for line in lines:
                    if '"insight"' in line.lower():
                        insight = line.split(':', 1)[-1].strip().strip('",')
                    elif '"takeaway"' in line.lower():
                        takeaway = line.split(':', 1)[-1].strip().strip('",')
                
                return {
                    "insight": insight or "This chart reveals important trends in the patent data.",
                    "takeaway": takeaway or "These patterns can guide future innovation strategies."
                }
                
        except Exception as e:
            logger.warning(f"Failed to generate dynamic insights: {e}")
            # Return fallback insights
            return {
                "insight": "This chart shows important patterns in the patent data worth analyzing further.",
                "takeaway": "Use these trends to inform strategic decisions about innovation and research priorities."
            }
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ChartGenerator:
    """Utility class for generating chart configurations"""
    
    @staticmethod
    def generate_line_chart(labels: List[str], data: List[int], title: str, label: str = "Patents") -> Dict[str, Any]:
        """Generate a line chart configuration"""
        return {
            'type': 'line',
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': label,
                    'data': data,
                    'borderColor': '#36A2EB',
                    'backgroundColor': 'rgba(54, 162, 235, 0.1)',
                    'fill': False,
                    'tension': 0.4,
                    'pointBackgroundColor': '#36A2EB',
                    'pointBorderColor': '#ffffff',
                    'pointBorderWidth': 2
                }]
            },
            'title': title
        }
    
    @staticmethod
    def generate_bar_chart(labels: List[str], data: List[int], title: str, label: str = "Patents") -> Dict[str, Any]:
        """Generate a bar chart configuration"""
        return {
            'type': 'bar',
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': label,
                    'data': data,
                    'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                    'borderColor': '#36A2EB',
                    'borderWidth': 1
                }]
            },
            'title': title
        }
    
    @staticmethod
    def generate_comparison_chart(labels: List[str], datasets: List[Dict], title: str) -> Dict[str, Any]:
        """Generate a comparison chart with multiple datasets"""
        colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FF9F40', '#9966FF']
        
        for i, dataset in enumerate(datasets):
            color = colors[i % len(colors)]
            dataset.update({
                'borderColor': color,
                'backgroundColor': f'rgba({ChartGenerator._hex_to_rgb(color)}, 0.1)',
                'fill': False,
                'tension': 0.4
            })
        
        return {
            'type': 'line',
            'data': {
                'labels': labels,
                'datasets': datasets
            },
            'title': title
        }
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to RGB string"""
        hex_color = hex_color.lstrip('#')
        return ', '.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))


class QueryResponse:
    """Standardized response format for all queries, always includes insight and takeaway."""
    
    def __init__(self, message: str, chart: Optional[Dict] = None, data: Optional[Dict] = None, insight: Optional[str] = "", takeaway: Optional[str] = ""):
        self.message = message
        self.chart = chart
        self.data = data
        self.insight = insight or ""
        self.takeaway = takeaway or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format, always includes insight and takeaway."""
        return {
            'message': self.message,
            'chart': self.chart,
            'data': self.data,
            'insight': self.insight,
            'takeaway': self.takeaway
        }


class DateUtils:
    """Utility functions for date operations"""
    
    @staticmethod
    def get_month_name(month: int) -> str:
        """Get month name from month number"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return months[month - 1] if 1 <= month <= 12 else f"Month{month}"
    
    @staticmethod
    def format_month_year(year: int, month: int) -> str:
        """Format year-month as string"""
        return f"{year}-{month:02d}"
