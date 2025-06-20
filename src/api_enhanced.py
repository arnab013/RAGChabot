#!/usr/bin/env python3
"""
Enhanced API handlers for different query types
"""
import re
try:
    from stats_queries_enhanced import PatentStatisticsEnhanced
except ImportError:
    from src.stats_queries_enhanced import PatentStatisticsEnhanced


def _handle_publication_trends_enhanced(query: str = "") -> dict:
    """Handle enhanced publication trends query with support for various query types"""
    query_lower = query.lower()
    
    # Parse different query types
    query_params = _parse_trends_query(query_lower)
    
    # Create enhanced stats instance
    stats_enhanced = PatentStatisticsEnhanced()
    
    try:
        # Call the enhanced function with parsed parameters
        data = stats_enhanced.get_publication_trends_enhanced(query_params)
        if not data:
            return {
                'message': "Sorry, I couldn't retrieve the publication trends.",
                'chart': None
            }
        
        # Generate response based on query type
        response = _format_trends_response(data, query_params)
        
        return {
            'message': "\n".join(response),
            'chart': data.get('chart')
        }
    
    finally:
        stats_enhanced.close()


def _parse_trends_query(query_lower: str) -> dict:
    """Parse different types of trends queries and return parameters"""
    
    # Initialize default parameters - FIXED: Default to all years instead of 12 months
    params = {
        'query_type': 'all_years',  # CHANGED: default to all years
        'months_back': 12,
        'years_back': 5,
        'specific_years': [],
        'comparison_years': [],
        'title_context': 'Publication Trends (All Available Data)'  # CHANGED: better default title
    }
    
    # Enhanced pattern matching
    month_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+months?', query_lower)
    year_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+years?', query_lower)
    # NEW: Flexible year pattern that doesn't require "last/past/recent"
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
    
    # NEW: Check for explicit "by year" pattern - this should show yearly aggregation for all data
    by_year_pattern = re.search(r'\bby\s+year\b', query_lower)
    
    # NEW: Check for patterns that suggest all-time view
    all_time_patterns = re.search(r'\b(?:all|total|entire|complete|full|overall)\s+(?:time|period|range|data|years?|history)\b', query_lower)
    
    # Check if any monthly pattern is present
    is_monthly_query = any(re.search(pattern, query_lower) for pattern in monthly_patterns) or monthly_in_year or monthly_for_year or monthly_last_year
    
    # Determine query type based on patterns - FIXED ORDER OF PRECEDENCE
    if monthly_last_year:
        # "by months in last year" - show last 12 months
        params['query_type'] = 'relative_months'
        params['months_back'] = 12
        params['title_context'] = "Publication Trends (Last 12 Months)"
        
    elif monthly_in_year or monthly_for_year:
        # "by month in 2023" - show monthly data for specific year
        year_match = monthly_in_year or monthly_for_year
        year = int(year_match.group(1))
        params['query_type'] = 'specific_years'
        params['specific_years'] = [year]
        params['title_context'] = f"Monthly Publication Trends for {year}"
        
    elif is_monthly_query and year_specific:
        # Monthly query with specific year mentioned
        params['query_type'] = 'specific_years'
        params['specific_years'] = [int(year) for year in year_specific]
        params['title_context'] = f"Monthly Publication Trends for {', '.join(year_specific)}"
        
    elif is_monthly_query and len(year_mentions) == 1:
        # Monthly query with year mentioned in text
        params['query_type'] = 'specific_years'
        params['specific_years'] = [int(year_mentions[0])]
        params['title_context'] = f"Monthly Publication Trends for {year_mentions[0]}"
        
    elif by_year_pattern and not flexible_year_match:
        # NEW: User explicitly wants yearly breakdown for all available data
        params['query_type'] = 'all_years'
        params['title_context'] = "Publication Trends by Year (All Available Data)"
    
    elif year_specific:
        params['query_type'] = 'specific_years'
        params['specific_years'] = [int(year) for year in year_specific]
        params['title_context'] = f"Publication Trends for {', '.join(year_specific)}"
    
    elif len(year_mentions) >= 2 and any(keyword in query_lower for keyword in comparison_keywords):
        params['query_type'] = 'comparison_years'
        params['comparison_years'] = [int(year) for year in sorted(set(year_mentions))]
        params['title_context'] = f"Publication Trends Comparison: {' vs '.join(map(str, params['comparison_years']))}"
    
    elif len(year_mentions) == 1:
        # Single year mentioned - treat as specific year
        params['query_type'] = 'specific_years'
        params['specific_years'] = [int(year_mentions[0])]
        params['title_context'] = f"Publication Trends for {year_mentions[0]}"
    
    elif month_match:
        params['query_type'] = 'relative_months'
        params['months_back'] = int(month_match.group(1))
        params['title_context'] = f"Publication Trends (Last {params['months_back']} Months)"
    
    elif year_match:
        params['query_type'] = 'relative_years'
        params['years_back'] = int(year_match.group(1))
        params['title_context'] = f"Publication Trends (Last {params['years_back']} Years)"
    
    elif flexible_year_match:
        # NEW: Handle patterns like "by 20 year" or "20 years"
        params['query_type'] = 'relative_years'
        params['years_back'] = int(flexible_year_match.group(1))
        params['title_context'] = f"Publication Trends (Last {params['years_back']} Years)"
    
    elif all_time_patterns:
        # NEW: Explicitly requested all-time view
        params['query_type'] = 'all_years'
        params['title_context'] = "Publication Trends (All Available Data)"
    
    # NEW: Default case for trend queries - show all available years instead of just 12 months
    elif 'trend' in query_lower or 'publication' in query_lower:
        params['query_type'] = 'all_years'
        params['title_context'] = "Publication Trends (All Available Data)"
    
    return params


def _format_trends_response(data: dict, query_params: dict) -> list:
    """Format the response text based on query type and data"""
    response = [f"**{query_params['title_context']}:**\n"]
    
    query_type = query_params['query_type']
    
    if query_type == 'relative_months' and data.get('monthly_complete'):
        response.append("📈 **Patents by Month:**")
        for month_data in data['monthly_complete']:
            year, month, count = month_data['year'], month_data['month'], month_data['count']
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]
            response.append(f"  • {month_name} {year}: {count:,} patents")
        
        total_monthly = sum(month_data['count'] for month_data in data['monthly_complete'])
        response.append(f"\n📅 **Total:** {total_monthly:,} patents")
        
    elif query_type in ['specific_years', 'comparison_years'] and data.get('yearly_monthly'):
        for year_data in data['yearly_monthly']:
            year = year_data['year']
            months = year_data['months']
            response.append(f"📈 **{year} Monthly Breakdown:**")
            
            for month_data in months:
                month, count = month_data['month'], month_data['count']
                month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]
                response.append(f"  • {month_name}: {count:,} patents")
            
            year_total = sum(month_data['count'] for month_data in months)
            response.append(f"  **{year} Total:** {year_total:,} patents\n")
            
    elif query_type == 'relative_years' and data.get('yearly'):
        response.append("📈 **Patents by Year:**")
        for year_data in data['yearly']:
            response.append(f"  • {year_data['year']}: {year_data['count']:,} patents")
        
        total_years = sum(year_data['count'] for year_data in data['yearly'])
        response.append(f"\n📅 **Total:** {total_years:,} patents")
    
    elif query_type == 'all_years' and data.get('yearly_complete'):
        response.append("📈 **Patents by Year:**")
        for year_data in data['yearly_complete']:
            response.append(f"  • {year_data['year']}: {year_data['count']:,} patents")
        
        total_years = sum(year_data['count'] for year_data in data['yearly_complete'])
        response.append(f"\n📅 **Total:** {total_years:,} patents")
        if data['yearly_complete']:
            first_year = data['yearly_complete'][0]['year']
            last_year = data['yearly_complete'][-1]['year']
            response.append(f"📅 **Years Covered:** {first_year} - {last_year}")

    return response


# Test function to verify the implementation
def test_enhanced_trends():
    """Test the enhanced trends functionality"""
    
    test_queries = [
        "Show me patent publication trends in last 12 months",
        "last 6 months patent trends",
        "publication trends in 2023",
        "trends for 2024",
        "compare 2023 and 2025 trends",
        "publication trends comparison 2023 vs 2024",
        "show trends for last 3 years"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing: '{query}' ===")
        result = _handle_publication_trends_enhanced(query)
        print(f"Message: {result['message'][:200]}...")
        if result['chart']:
            print(f"Chart type: {result['chart']['type']}")
            print(f"Chart title: {result['chart']['title']}")


if __name__ == "__main__":
    test_enhanced_trends()
