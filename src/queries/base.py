"""
Base classes and utilities for query handlers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from database.config import get_db_session_simple
from database.models import Patent
import logging

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
    """Standardized response format for all queries"""
    
    def __init__(self, message: str, chart: Optional[Dict] = None, data: Optional[Dict] = None):
        self.message = message
        self.chart = chart
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'message': self.message,
            'chart': self.chart,
            'data': self.data
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
