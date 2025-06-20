"""
Handler for publication trends queries
Supports various query types:
- Relative time: "last 12 months", "past 5 years"  
- Specific years: "trends in 2023", "publication trends for 2024"
- Comparisons: "compare 2023 and 2025", "2023 vs 2024 trends"
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sqlalchemy import func, extract
from .base import BaseQueryHandler, ChartGenerator, QueryResponse, DateUtils
from database.models import Patent


class PublicationTrendsHandler(BaseQueryHandler):
    """Handler for publication trends queries"""
    
    def get_query_keywords(self) -> List[str]:
        """Keywords that identify publication trends queries"""
        return [
            "publication trend", "publication trends", "trends", "trend",
            "monthly", "yearly", "annual", "by year", "by month",
            "last", "past", "recent", "compare", "comparison", "vs", "versus"
        ]
    
    def handle_query(self, query: str, **kwargs) -> QueryResponse:
        """Handle publication trends query"""
        try:
            # Parse the query to determine type and parameters
            query_params = self._parse_query(query.lower())
            
            # Route to appropriate handler based on query type
            if query_params['type'] == 'relative_months':
                return self._handle_relative_months(query, query_params)
            elif query_params['type'] == 'relative_years':
                return self._handle_relative_years(query, query_params)
            elif query_params['type'] == 'specific_years':
                return self._handle_specific_years(query, query_params)
            elif query_params['type'] == 'comparison_years':
                return self._handle_comparison_years(query, query_params)
            elif query_params['type'] == 'all_years':
                return self._handle_all_years(query, query_params)
            else:
                # Default to all years
                return self._handle_all_years(query, {'title': 'Publication Trends (All Available Data)'})
                
        except Exception as e:
            try:
                error_message = self.generate_error_message(
                    query=query,
                    error_type="publication_trends_error",
                    technical_error=str(e)
                )
            except Exception:
                # Fallback if error message generation fails
                error_message = "I'm having trouble generating publication trends at the moment. Please try a different query or try again later."
            return QueryResponse(message=error_message)
    
    def _parse_query(self, query_lower: str) -> Dict[str, Any]:
        """Parse query to extract parameters"""
        params = {'type': 'all_years', 'title': 'Publication Trends'}  # Changed default to all years
        
        # Enhanced pattern matching
        month_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+months?', query_lower)
        year_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+years?', query_lower)
        # New: flexible year pattern that doesn't require "last/past/recent"
        flexible_year_match = re.search(r'(?:by\s+)?(\d+)\s+years?(?:\s+(?:period|span|range))?', query_lower)
        year_specific = re.findall(r'(?:in|for|during)\s+(\d{4})', query_lower)
        year_mentions = re.findall(r'\b(20\d{2})\b', query_lower)
        comparison_keywords = ['compar', 'vs', 'versus', 'against', 'and']
        
        # NEW: Monthly pattern detection
        monthly_patterns = [
            r'\bby\s+months?\b',
            r'\bmonthly\b',
            r'\bper\s+month\b',
            r'\bmonth\s+by\s+month\b'
        ]
        
        # NEW: Monthly in specific year pattern
        monthly_in_year = re.search(r'(?:by\s+)?months?\s+(?:in|for|during)\s+(\d{4})', query_lower)
        monthly_for_year = re.search(r'monthly.*?(?:in|for|during)\s+(\d{4})', query_lower)
        
        # NEW: Monthly in last year pattern
        monthly_last_year = re.search(r'(?:by\s+)?months?\s+(?:in\s+)?(?:last|past|recent)\s+year', query_lower)
        
        # Check for explicit "by year" pattern - this should show yearly aggregation
        by_year_pattern = re.search(r'\bby\s+year\b', query_lower)
        
        # Check for patterns that suggest all-time view
        all_time_patterns = re.search(r'\b(?:all|total|entire|complete|full|overall)\s+(?:time|period|range|data|years?|history)\b', query_lower)
        
        # Check if any monthly pattern is present
        is_monthly_query = any(re.search(pattern, query_lower) for pattern in monthly_patterns) or monthly_in_year or monthly_for_year or monthly_last_year
        
        # Determine query type based on patterns - ENHANCED ORDER OF PRECEDENCE
        if monthly_last_year:
            # "by months in last year" - show last 12 months
            params['type'] = 'relative_months'
            params['months'] = 12
            params['title'] = "Publication Trends (Last 12 Months)"
            
        elif monthly_in_year or monthly_for_year:
            # "by month in 2023" - show monthly data for specific year
            year_match = monthly_in_year or monthly_for_year
            year = int(year_match.group(1))
            params['type'] = 'specific_years'
            params['years'] = [year]
            params['title'] = f"Monthly Publication Trends for {year}"
            
        elif is_monthly_query and year_specific:
            # Monthly query with specific year mentioned
            params['type'] = 'specific_years'
            params['years'] = [int(year) for year in year_specific]
            params['title'] = f"Monthly Publication Trends for {', '.join(year_specific)}"
            
        elif is_monthly_query and len(year_mentions) == 1:
            # Monthly query with year mentioned in text
            params['type'] = 'specific_years'
            params['years'] = [int(year_mentions[0])]
            params['title'] = f"Monthly Publication Trends for {year_mentions[0]}"
        
        elif by_year_pattern and not flexible_year_match:
            # User explicitly wants yearly breakdown for all available data
            params['type'] = 'all_years'
            params['title'] = "Publication Trends by Year (All Available Data)"
        
        elif year_specific:
            params['type'] = 'specific_years'
            params['years'] = [int(year) for year in year_specific]
            params['title'] = f"Publication Trends for {', '.join(year_specific)}"
        
        elif len(year_mentions) >= 2 and any(keyword in query_lower for keyword in comparison_keywords):
            params['type'] = 'comparison_years'
            params['years'] = [int(year) for year in sorted(set(year_mentions))]
            params['title'] = f"Publication Trends Comparison: {' vs '.join(map(str, params['years']))}"
        
        elif len(year_mentions) == 1:
            params['type'] = 'specific_years'
            params['years'] = [int(year_mentions[0])]
            params['title'] = f"Publication Trends for {year_mentions[0]}"
        
        elif month_match:
            params['type'] = 'relative_months'
            params['months_back'] = int(month_match.group(1))
            params['title'] = f"Publication Trends (Last {params['months_back']} Months)"
        
        elif year_match:
            params['type'] = 'relative_years'
            params['years_back'] = int(year_match.group(1))
            params['title'] = f"Publication Trends (Last {params['years_back']} Years)"
        
        elif flexible_year_match:
            # Handle patterns like "by 20 year" or "20 years"
            params['type'] = 'relative_years'
            params['years_back'] = int(flexible_year_match.group(1))
            params['title'] = f"Publication Trends (Last {params['years_back']} Years)"
        
        elif all_time_patterns:
            # Explicitly requested all-time view
            params['type'] = 'all_years'
            params['title'] = "Publication Trends (All Available Data)"
        
        # Default case: if no specific time pattern detected, show all available years
        # This fixes the issue where generic queries defaulted to 12 months
        elif 'trend' in query_lower or 'publication' in query_lower:
            params['type'] = 'all_years'
            params['title'] = "Publication Trends (All Available Data)"
        
        return params
    
    def _handle_relative_months(self, query: str, params: Dict[str, Any]) -> QueryResponse:
        """Handle relative months queries like 'last 12 months'"""
        months_back = params.get('months_back', 12)
        title = params.get('title', f'Last {months_back} Months')
        
        # Calculate start date
        current_date = datetime.now()
        start_date = datetime(current_date.year, current_date.month, 1)
        
        # Go back the required number of months
        for _ in range(months_back - 1):
            if start_date.month == 1:
                start_date = start_date.replace(year=start_date.year - 1, month=12)
            else:
                start_date = start_date.replace(month=start_date.month - 1)
        
        # Get data from database
        monthly_stats = self.session.query(
            extract('year', Patent.publication_date).label('year'),
            extract('month', Patent.publication_date).label('month'),
            func.count(Patent.publication_number).label('count')
        ).filter(
            Patent.publication_date.isnot(None),
            Patent.publication_date >= start_date
        ).group_by(
            extract('year', Patent.publication_date),
            extract('month', Patent.publication_date)
        ).order_by('year', 'month').all()
        
        # Create lookup dict
        data_dict = {(int(year), int(month)): count for year, month, count in monthly_stats if year is not None and month is not None}
        
        # Generate complete month series
        complete_months = []
        current = start_date
        
        for i in range(months_back):
            year, month = current.year, current.month
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
        
        # Generate response
        response_lines = [f"**{title}:**\n", "**Patents by Month:**"]
        
        for month_data in complete_months:
            year, month, count = month_data['year'], month_data['month'], month_data['count']
            month_name = DateUtils.get_month_name(month)
            response_lines.append(f"  • {month_name} {year}: {count:,} patents")
        
        total = sum(m['count'] for m in complete_months)
        response_lines.append(f"\n**Total:** {total:,} patents")
        
        # Generate chart
        labels = [f"{DateUtils.get_month_name(m['month'])} {m['year']}" for m in complete_months]
        values = [m['count'] for m in complete_months]
        chart = ChartGenerator.generate_line_chart(labels, values, title)
        
        # Generate dynamic insights using LLM
        data_summary = f"Monthly patent publications from {complete_months[0]['year']}-{complete_months[0]['month']:02d} to {complete_months[-1]['year']}-{complete_months[-1]['month']:02d}. Total: {total:,} patents. Peak: {max(values)} patents, Low: {min(values)} patents."
        insights = self.generate_dynamic_insights(query, chart, data_summary)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'monthly_complete': complete_months},
            insight=insights["insight"],
            takeaway=insights["takeaway"]
        )
    
    def _handle_relative_years(self, query: str, params: Dict[str, Any]) -> QueryResponse:
        """Handle relative years queries like 'last 5 years'"""
        years_back = params.get('years_back', 5)
        title = params.get('title', f'Last {years_back} Years')
        
        current_year = datetime.now().year
        start_year = current_year - years_back + 1
        
        # Get data from database
        yearly_stats = self.session.query(
            extract('year', Patent.publication_date).label('year'),
            func.count(Patent.publication_number).label('count')
        ).filter(
            Patent.publication_date.isnot(None),
            extract('year', Patent.publication_date) >= start_year
        ).group_by(
            extract('year', Patent.publication_date)
        ).order_by('year').all()
        
        # Create lookup dict and complete series
        data_dict = {int(year): count for year, count in yearly_stats if year is not None}
        complete_years = []
        
        for year in range(start_year, current_year + 1):
            complete_years.append({
                'year': year,
                'count': data_dict.get(year, 0)
            })
        
        # Generate response
        response_lines = [f"**{title}:**\n", "**Patents by Year:**"]
        
        for year_data in complete_years:
            response_lines.append(f"  • {year_data['year']}: {year_data['count']:,} patents")
        
        total = sum(y['count'] for y in complete_years)
        response_lines.append(f"\n**Total:** {total:,} patents")
        
        # Generate chart
        labels = [str(y['year']) for y in complete_years]
        values = [y['count'] for y in complete_years]
        chart = ChartGenerator.generate_line_chart(labels, values, title)
        
        # Generate dynamic insights using LLM
        data_summary = f"Yearly patent publications from {start_year} to {current_year}. Total: {total:,} patents. Peak: {max(values)} patents in {labels[values.index(max(values))]}, Low: {min(values)} patents in {labels[values.index(min(values))]}."
        insights = self.generate_dynamic_insights(query, chart, data_summary)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'yearly_complete': complete_years},
            insight=insights["insight"],
            takeaway=insights["takeaway"]
        )
    
    def _handle_specific_years(self, query: str, params: Dict[str, Any]) -> QueryResponse:
        """Handle specific year queries like 'trends in 2023'"""
        years = params.get('years', [2023])
        title = params.get('title', f'Publication Trends for {", ".join(map(str, years))}')
        
        yearly_monthly = []
        
        for year in years:
            # Get monthly data for this year
            monthly_stats = self.session.query(
                extract('month', Patent.publication_date).label('month'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date.isnot(None),
                extract('year', Patent.publication_date) == year
            ).group_by(
                extract('month', Patent.publication_date)
            ).order_by('month').all()
            
            # Create complete 12-month series
            data_dict = {int(month): count for month, count in monthly_stats if month is not None}
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
        
        # Generate response
        response_lines = [f"**{title}:**\n"]
        
        for year_data in yearly_monthly:
            year = year_data['year']
            months = year_data['months']
            response_lines.append(f"**{year} Monthly Breakdown:**")
            
            for month_data in months:
                month, count = month_data['month'], month_data['count']
                month_name = DateUtils.get_month_name(month)
                response_lines.append(f"  • {month_name}: {count:,} patents")
            
            year_total = sum(m['count'] for m in months)
            response_lines.append(f"  **{year} Total:** {year_total:,} patents\n")
        
        # Generate chart
        if len(years) == 1:
            # Single year chart
            year_data = yearly_monthly[0]
            labels = [DateUtils.get_month_name(m['month']) for m in year_data['months']]
            values = [m['count'] for m in year_data['months']]
            chart = ChartGenerator.generate_line_chart(labels, values, title)
        else:
            # Multiple years - we'll handle this in comparison
            chart = self._generate_multi_year_chart(yearly_monthly, title)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'yearly_monthly': yearly_monthly}
        )
    
    def _handle_comparison_years(self, query: str, params: Dict[str, Any]) -> QueryResponse:
        """Handle comparison queries like '2023 vs 2025'"""
        years = params.get('years', [2023, 2025])
        title = params.get('title', f'Publication Trends Comparison: {" vs ".join(map(str, years))}')
        
        # Get data for each year (reuse specific years logic)
        yearly_monthly = []
        
        for year in years:
            monthly_stats = self.session.query(
                extract('month', Patent.publication_date).label('month'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date.isnot(None),
                extract('year', Patent.publication_date) == year
            ).group_by(
                extract('month', Patent.publication_date)
            ).order_by('month').all()
            
            data_dict = {int(month): count for month, count in monthly_stats if month is not None}
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
        
        # Generate response (same as specific years)
        response_lines = [f"**{title}:**\n"]
        
        for year_data in yearly_monthly:
            year = year_data['year']
            months = year_data['months']
            response_lines.append(f"**{year} Monthly Breakdown:**")
            
            for month_data in months:
                month, count = month_data['month'], month_data['count']
                month_name = DateUtils.get_month_name(month)
                response_lines.append(f"  • {month_name}: {count:,} patents")
            
            year_total = sum(m['count'] for m in months)
            response_lines.append(f"  **{year} Total:** {year_total:,} patents\n")
        
        # Generate comparison chart
        chart = self._generate_multi_year_chart(yearly_monthly, title)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'yearly_monthly': yearly_monthly}
        )
    
    def _generate_multi_year_chart(self, yearly_monthly: List[Dict], title: str) -> Dict[str, Any]:
        """Generate chart for multiple years comparison"""
        labels = [DateUtils.get_month_name(m) for m in range(1, 13)]
        datasets = []
        
        for year_data in yearly_monthly:
            year = year_data['year']
            values = [m['count'] for m in year_data['months']]
            
            datasets.append({
                'label': str(year),
                'data': values
            })
        
        return ChartGenerator.generate_comparison_chart(labels, datasets, title)

    def _handle_all_years(self, query: str, params: Dict[str, Any]) -> QueryResponse:
        """Handle all years queries like 'by year' or generic 'publication trends'"""
        title = params.get('title', 'Publication Trends (All Available Data)')
        
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
            if year is not None:  # Extra safety check
                yearly_data.append({
                    'year': int(year),
                    'count': count
                })
        
        if not yearly_data:
            return QueryResponse(
                message="No publication data found in the database.",
                chart=None,
                data={'yearly_data': []}
            )
        
        # Generate response
        response_lines = [f"**{title}:**\n", "**Patents by Year:**"]
        
        for year_data in yearly_data:
            response_lines.append(f"  • {year_data['year']}: {year_data['count']:,} patents")
        
        total = sum(y['count'] for y in yearly_data)
        response_lines.append(f"\n**Total:** {total:,} patents")
        response_lines.append(f"**Years Covered:** {yearly_data[0]['year']} - {yearly_data[-1]['year']}")
        
        # Generate chart
        labels = [str(y['year']) for y in yearly_data]
        values = [y['count'] for y in yearly_data]
        chart = ChartGenerator.generate_line_chart(labels, values, title)
        
        # Generate dynamic insights using LLM
        year_range = f"{yearly_data[0]['year']}-{yearly_data[-1]['year']}"
        peak_year = yearly_data[values.index(max(values))]['year'] if values else "N/A"
        low_year = yearly_data[values.index(min(values))]['year'] if values else "N/A"
        data_summary = f"Patent publications from {year_range}. Total: {total:,} patents across {len(yearly_data)} years. Peak: {max(values) if values else 0} patents in {peak_year}, Low: {min(values) if values else 0} patents in {low_year}."
        insights = self.generate_dynamic_insights(query, chart, data_summary)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'yearly_data': yearly_data},
            insight=insights["insight"],
            takeaway=insights["takeaway"]
        )
