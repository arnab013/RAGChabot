#!/usr/bin/env python3
"""
Enhanced API handlers for different query types
"""
import re
from stats_queries_enhanced import PatentStatisticsEnhanced


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
    
    # Initialize default parameters
    params = {
        'query_type': 'relative_months',  # default
        'months_back': 12,
        'specific_years': [],
        'comparison_years': [],
        'title_context': 'Publication Trends'
    }
    
    # Pattern 1: Relative time periods (last X months/years)
    month_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+months?', query_lower)
    year_match = re.search(r'(?:last|past|recent)\s+(\d+)\s+years?', query_lower)
    
    # Pattern 2: Specific years (trends in 2023, trends for 2024)
    year_specific = re.findall(r'(?:in|for|during)\s+(\d{4})', query_lower)
    
    # Pattern 3: Year mentions for comparison (2023 and 2024, compare 2023 vs 2025)
    year_mentions = re.findall(r'\b(20\d{2})\b', query_lower)
    comparison_keywords = ['compar', 'vs', 'versus', 'against', 'and']
    
    # Determine query type based on patterns
    if year_specific:
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
