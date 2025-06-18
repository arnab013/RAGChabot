"""
Queries package - Modular query handlers for patent data analysis
"""

from .query_manager import QueryManager, handle_query
from .base import BaseQueryHandler, QueryResponse, ChartGenerator, DateUtils
from .publication_trends import PublicationTrendsHandler
from .sdg_distribution import SDGDistributionHandler
from .technology_analysis import TechnologyAnalysisHandler
from .inventor_assignee import InventorAssigneeHandler
from .geographical_analysis import GeographicalAnalysisHandler

__all__ = [
    'QueryManager',
    'handle_query',
    'BaseQueryHandler',
    'QueryResponse',
    'ChartGenerator',
    'DateUtils',
    'PublicationTrendsHandler',
    'SDGDistributionHandler',
    'TechnologyAnalysisHandler',
    'InventorAssigneeHandler',
    'GeographicalAnalysisHandler'
]
