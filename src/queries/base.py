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
    from ..llm_clients import chat
except ImportError:
    try:
        from llm_clients import chat
    except ImportError:
        chat = None

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
            
            # Construct LLM prompt with formatting instructions
            prompt = f"""Based on the following patent data analysis, generate concise and insightful insights and takeaways.

Query: {query}
Chart Type: {chart_type}
Chart Title: {chart_title}
Data Summary: {data_summary}
Key Data Points: {', '.join(data_points[:8])}

FORMATTING REQUIREMENTS:
- Use **bold text** for key terms and numbers
- Include bullet points where appropriate
- Make insights specific and actionable
- Use clear, well-structured language

Please provide:
1. A brief insight (2-3 sentences) highlighting the most important trend or pattern in the data
2. A practical takeaway (2-3 sentences) explaining what this means for innovation strategy or business decisions

Format your response as JSON:
{{
    "insight": "Your well-formatted insight with **bold text** for emphasis and • bullet points if needed",
    "takeaway": "Your well-formatted takeaway with **bold text** and clear structure"
}}"""

            messages = [{"role": "user", "content": prompt}]
              # Call LLM
            response = chat(messages, temperature=0.3, max_tokens=300)
            
            # Parse LLM response
            import json
            
            try:
                parsed_response = json.loads(response)
                return {
                    "insight": parsed_response.get("insight", "The analysis reveals **significant patterns** in patent activity that warrant further examination."),
                    "takeaway": parsed_response.get("takeaway", "Organizations can leverage these insights to optimize their **innovation strategies** and competitive positioning.")
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
                    "insight": insight or "The data reveals **meaningful trends** that demonstrate innovation patterns across the patent landscape.",
                    "takeaway": takeaway or "These analytical findings provide **actionable intelligence** for strategic research and development planning."
                }
                
        except Exception as e:
            logger.warning(f"Failed to generate dynamic insights: {e}")
            # Return fallback insights with formatting
            return {
                "insight": "The patent analysis reveals **significant trends** that demonstrate important innovation patterns across the technology landscape.",
                "takeaway": "These insights provide **valuable intelligence** for strategic research planning and competitive positioning in key innovation areas."
            }

    def generate_error_message(self, query: str, error_type: str, technical_error: str) -> str:
        """Generate user-friendly error message using LLM if available"""
        try:
            # Import here to avoid circular imports
            from ..llm_clients import chat
            
            prompt = f"""
A user asked: "{query}"

The patent analytics system encountered an error ({error_type}): {technical_error}

Generate a helpful, user-friendly message that:
1. Acknowledges their request
2. Explains that there was an issue processing the data
3. Suggests they try a different approach or query
4. Maintains a professional and helpful tone

Keep it concise (2-3 sentences) and avoid technical details.
"""
            
            messages = [{"role": "user", "content": prompt}]
            error_message = chat(messages, temperature=0.7, max_tokens=150)
            return error_message.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate dynamic error message: {e}")
            # Fallback to static error message when LLM is unavailable
            return "I'm unable to process your query at the moment. Please try again later or try a different type of query."

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
                'datasets': [{                    'label': label,
                    'data': data,
                    'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                    'borderColor': '#36A2EB',
                    'borderWidth': 1
                }]
            },
            'title': title
        }
    
    @staticmethod
    def generate_horizontal_bar_chart(labels: List[str], data: List[int], title: str, label: str = "Patents") -> Dict[str, Any]:
        """Generate a horizontal bar chart configuration"""
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
    def generate_pie_chart(labels: List[str], data: List[int], title: str) -> Dict[str, Any]:
        """Generate a pie chart configuration"""
        return {
            'type': 'pie',
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': data,
                    'backgroundColor': [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
                        '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56',
                        '#9966FF', '#FF9F40', '#4BC0C0', '#C9CBCF', '#FF6384'
                    ]
                }]
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
