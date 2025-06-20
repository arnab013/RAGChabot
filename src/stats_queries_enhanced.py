#!/usr/bin/env python3
"""
Enhanced stats_queries.py with support for different query types
"""
import json
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sqlalchemy import func, distinct, extract, text
from sqlalchemy.orm import sessionmaker
from database.models import Patent, PatentChunk
from database.config import get_db_session, get_db_session_simple

logger = logging.getLogger(__name__)

class PatentStatisticsEnhanced:
    """Enhanced patent statistics with support for various query types"""
    def __init__(self):
        self.session = get_db_session_simple()
    
    def get_publication_trends_enhanced(self, query_params: dict):
        """Get enhanced publication trends based on query parameters"""
        try:
            query_type = query_params['query_type']
            
            if query_type == 'relative_months':
                return self._get_relative_months_data(query_params)
            elif query_type == 'specific_years':
                return self._get_specific_years_data(query_params)
            elif query_type == 'comparison_years':
                return self._get_comparison_years_data(query_params)
            elif query_type == 'relative_years':
                return self._get_relative_years_data(query_params)
            elif query_type == 'all_years':
                return self._get_all_years_data(query_params)
            else:
                # Default to all years if type is not recognized
                return self._get_all_years_data(query_params)
                
        except Exception as e:
            logger.error(f"Error getting enhanced publication trends: {e}")
            return None
    
    def _get_relative_months_data(self, query_params: dict):
        """Get data for relative months queries (e.g., 'last 12 months')"""
        months_back = query_params['months_back']
        
        # Calculate the start date - go back to the first day of the start month
        current_date = datetime.now()
        start_date = datetime(current_date.year, current_date.month, 1)
        
        # Go back the required number of months
        for _ in range(months_back - 1):
            if start_date.month == 1:
                start_date = start_date.replace(year=start_date.year - 1, month=12)
            else:
                start_date = start_date.replace(month=start_date.month - 1)
        
        # Get actual data from database
        monthly_stats = self.session.query(
            extract('year', Patent.publication_date).label('year'),
            extract('month', Patent.publication_date).label('month'),
            func.count(Patent.publication_number).label('count')
        ).filter(
            Patent.publication_date >= start_date
        ).group_by(
            extract('year', Patent.publication_date),
            extract('month', Patent.publication_date)
        ).order_by('year', 'month').all()
        
        # Convert to dict for easy lookup
        data_dict = {(int(year), int(month)): count for year, month, count in monthly_stats}
        
        # Generate complete month series with zeros for missing months
        complete_months = []
        current = start_date
        
        for i in range(months_back):
            year = current.year
            month = current.month
            count = data_dict.get((year, month), 0)
            
            complete_months.append({
                'year': year,
                'month': month,
                'count': count
            })
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Generate chart data
        chart_data = [
            {
                'category': f"{item['year']}-{item['month']:02d}",
                'value': item['count']
            }
            for item in complete_months
        ]
        
        chart = self._generate_chart(chart_data, 'line', query_params['title_context'])
        
        return {
            'monthly_complete': complete_months,
            'chart': chart
        }
    
    def _get_specific_years_data(self, query_params: dict):
        """Get data for specific years queries (e.g., 'trends in 2023')"""
        years = query_params['specific_years']
        yearly_monthly = []
        
        for year in years:
            # Get monthly data for this specific year
            monthly_stats = self.session.query(
                extract('month', Patent.publication_date).label('month'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                extract('year', Patent.publication_date) == year
            ).group_by(
                extract('month', Patent.publication_date)
            ).order_by('month').all()
            
            # Convert to dict for easy lookup
            data_dict = {int(month): count for month, count in monthly_stats}
            
            # Generate complete 12-month series
            months = []
            for month in range(1, 13):
                months.append({
                    'month': month,
                    'count': data_dict.get(month, 0)
                })
            
            yearly_monthly.append({
                'year': year,
                'months': months
            })
        
        # Generate chart data
        if len(years) == 1:
            # Single year - simple monthly chart
            chart_data = [
                {
                    'category': f"{years[0]}-{month['month']:02d}",
                    'value': month['count']
                }
                for month in yearly_monthly[0]['months']
            ]
            chart_type = 'line'
        else:
            # Multiple years - comparison chart (we'll handle this later)
            chart_data = []
            chart_type = 'line'
        
        chart = self._generate_chart(chart_data, chart_type, query_params['title_context'])
        
        return {
            'yearly_monthly': yearly_monthly,
            'chart': chart
        }
    
    def _get_comparison_years_data(self, query_params: dict):
        """Get data for comparison years queries (e.g., '2023 vs 2025')"""
        years = query_params['comparison_years']
        yearly_monthly = []
        
        # Get data for each year
        for year in years:
            monthly_stats = self.session.query(
                extract('month', Patent.publication_date).label('month'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                extract('year', Patent.publication_date) == year
            ).group_by(
                extract('month', Patent.publication_date)
            ).order_by('month').all()
            
            data_dict = {int(month): count for month, count in monthly_stats}
            
            months = []
            for month in range(1, 13):
                months.append({
                    'month': month,
                    'count': data_dict.get(month, 0)
                })
            
            yearly_monthly.append({
                'year': year,
                'months': months
            })
        
        # Generate comparison chart data
        chart_data = self._generate_comparison_chart_data(yearly_monthly)
        chart = self._generate_comparison_chart(chart_data, query_params['title_context'])
        
        return {
            'yearly_monthly': yearly_monthly,
            'chart': chart
        }
    
    def _get_relative_years_data(self, query_params: dict):
        """Get data for relative years queries (e.g., 'last 5 years')"""
        years_back = query_params.get('years_back', 5)
        
        current_year = datetime.now().year
        start_year = current_year - years_back + 1
        
        yearly_stats = self.session.query(
            extract('year', Patent.publication_date).label('year'),
            func.count(Patent.publication_number).label('count')
        ).filter(
            extract('year', Patent.publication_date) >= start_year
        ).group_by(
            extract('year', Patent.publication_date)
        ).order_by('year').all()
        
        # Convert to dict and fill missing years
        data_dict = {int(year): count for year, count in yearly_stats}
        
        yearly_complete = []
        for year in range(start_year, current_year + 1):
            yearly_complete.append({
                'year': year,
                'count': data_dict.get(year, 0)
            })
        
        # Generate chart
        chart_data = [
            {
                'category': str(item['year']),
                'value': item['count']
            }
            for item in yearly_complete
        ]
        
        chart = self._generate_chart(chart_data, 'line', query_params['title_context'])
        
        return {
            'yearly': yearly_complete,
            'chart': chart
        }
    
    def _get_all_years_data(self, query_params: dict):
        """Get data for all years queries (e.g., 'by year' or generic 'publication trends')"""
        from sqlalchemy import extract, func
        from database.models import Patent
        
        title = query_params.get('title_context', 'Publication Trends (All Available Data)')
        
        try:
            # Get all years with data from database
            yearly_stats = self.session.query(
                extract('year', Patent.publication_date).label('year'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date.isnot(None)
            ).group_by(
                extract('year', Patent.publication_date)
            ).order_by('year').all()
            
            # Convert to list of dicts
            yearly_data = []
            for year, count in yearly_stats:
                if year is not None:
                    yearly_data.append({
                        'year': int(year),
                        'count': count
                    })
            
            if not yearly_data:
                return {
                    'yearly_complete': [],
                    'chart': None
                }
            
            # Generate chart
            labels = [str(y['year']) for y in yearly_data]
            values = [y['count'] for y in yearly_data]
            
            chart = {
                'type': 'line',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Patents Published',
                        'data': values,
                        'borderColor': 'rgb(75, 192, 192)',
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'tension': 0.1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': title
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Number of Patents'
                            }
                        },
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Year'
                            }
                        }
                    }
                }
            }
            
            return {
                'yearly_complete': yearly_data,
                'chart': chart
            }
            
        except Exception as e:
            logger.error(f"Error getting all years data: {e}")
            return None
    
    def _generate_chart(self, chart_data: list, chart_type: str, title: str):
        """Generate chart configuration"""
        if not chart_data:
            return None
        
        labels = [item['category'] for item in chart_data]
        values = [item['value'] for item in chart_data]
        
        return {
            'type': chart_type,
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': 'Patents',
                    'data': values,
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
    
    def _generate_comparison_chart_data(self, yearly_monthly: list):
        """Generate chart data for year comparisons"""
        chart_data = []
        
        # Create datasets for each year
        datasets = []
        colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FF9F40', '#9966FF']
        
        for i, year_data in enumerate(yearly_monthly):
            year = year_data['year']
            values = [month['count'] for month in year_data['months']]
            
            datasets.append({
                'label': f'{year}',
                'data': values,
                'borderColor': colors[i % len(colors)],
                'backgroundColor': f'rgba({self._hex_to_rgb(colors[i % len(colors)])}, 0.1)',
                'fill': False,
                'tension': 0.4
            })
        
        return {
            'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'datasets': datasets
        }
    
    def _generate_comparison_chart(self, chart_data: dict, title: str):
        """Generate comparison chart configuration"""
        return {
            'type': 'line',
            'data': chart_data,
            'title': title
        }
    
    def _hex_to_rgb(self, hex_color: str):
        """Convert hex color to RGB string"""
        hex_color = hex_color.lstrip('#')
        return ', '.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
    
    def close(self):
        """Close the database session"""
        if self.session:
            self.session.close()
