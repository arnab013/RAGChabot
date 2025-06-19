"""
Handler for geographical analysis queries
"""
import json
from collections import Counter
from typing import Dict, List, Any
from .base import BaseQueryHandler, ChartGenerator, QueryResponse
from database.models import Patent


class GeographicalAnalysisHandler(BaseQueryHandler):
    """Handler for geographical analysis queries"""
    
    def get_query_keywords(self) -> List[str]:
        """Keywords that identify geographical queries"""
        return [
            "country", "countries", "region", "geographical", "location",
            "origin", "priority", "nationality", "global", "international"
        ]
    
    def handle_query(self, query: str, **kwargs) -> QueryResponse:
        """Handle geographical analysis query"""
        try:
            query_lower = query.lower()
            
            # Extract limit from query
            import re
            limit_match = re.search(r'(?:top|first|leading)\s+(\d+)', query_lower)
            limit = int(limit_match.group(1)) if limit_match else 10
            
            # Get geographical data based on applicant countries
            patents_with_countries = self.session.query(Patent.applicant_countries).filter(
                Patent.applicant_countries.isnot(None),
                Patent.applicant_countries != ''
            ).all()
            
            country_counter = Counter()
            
            for patent in patents_with_countries:
                if patent.applicant_countries:
                    try:
                        # Parse JSON array of countries
                        countries = json.loads(patent.applicant_countries)
                        if isinstance(countries, list):
                            for country in countries:
                                if isinstance(country, str) and country.strip():
                                    country_counter[country.strip()] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # Get top countries
            top_countries = country_counter.most_common(limit)
            
            # Format data
            country_data = [
                {'country': country, 'count': count}
                for country, count in top_countries
            ]
            
            # Generate response
            response_lines = self._format_geographical_response(country_data, limit)
            
            # Generate chart
            if country_data:
                labels = [item['country'] for item in country_data]
                values = [item['count'] for item in country_data]
                chart = ChartGenerator.generate_bar_chart(labels, values, f"Top {limit} Countries by Patent Count")
            else:
                chart = None
            
            # Generate simple insight and takeaway
            insight = "This chart shows the top countries by patent filings."
            takeaway = "Identify which countries are leading in innovation based on patent activity."
            return QueryResponse(
                message="\n".join(response_lines),
                chart=chart,
                data={'country_stats': country_data},
                insight=insight,
                takeaway=takeaway
            )
            
        except Exception as e:
            return QueryResponse(
                message=f"Sorry, I couldn't retrieve the geographical analysis. Error: {str(e)}"
            )
    
    def _format_geographical_response(self, country_data: List[Dict], limit: int) -> List[str]:
        """Format geographical analysis response"""
        total_patents = sum(item['count'] for item in country_data)
        
        response_lines = [
            f"**Top {limit} Countries by Applicant Origin:**\n",
            f"📊 **Total Patents from Top Countries:** {total_patents:,}\n",
            "🌍 **Geographical Distribution:**"
        ]
        
        for i, item in enumerate(country_data, 1):
            country = item['country']
            count = item['count']
            percentage = (count / total_patents * 100) if total_patents > 0 else 0
            response_lines.append(f"  {i}. {country}: {count:,} patents ({percentage:.1f}%)")
        
        return response_lines
