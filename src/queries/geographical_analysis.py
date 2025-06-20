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
            "country", "countries", "region", "geographical", "geographical distribution", 
            "location", "origin", "priority", "nationality", "global", "international",
            "by country", "per country", "country wise", "countrywise"
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
            ]            # Generate response
            response_lines = self._format_geographical_response(country_data, limit)
            # Generate chart
            if country_data:
                labels = [item['country'] for item in country_data]
                values = [item['count'] for item in country_data]
                chart = ChartGenerator.generate_horizontal_bar_chart(labels, values, f"Top {limit} Countries by Patent Count")
            else:
                chart = None
            
            # Generate dynamic insights and takeaway
            data_summary = f"Top {limit} countries by patent count. Data shows {len(country_data)} countries with their patent filing counts."
            insights = self.generate_dynamic_insights(query_lower, chart or {}, data_summary)
            
            return QueryResponse(
                message="\n".join(response_lines),
                chart=chart,
                data={'country_stats': country_data},
                insight=insights['insight'],
                takeaway=insights['takeaway']
            )
        except Exception as e:            # Use dynamic LLM-based error message generation
            try:
                from ..llm_clients import chat
                
                prompt = f"""
A user asked: "{query}"

The patent analytics system encountered an error while analyzing geographical data: {str(e)}

Generate a helpful, user-friendly message that:
1. Acknowledges their request for geographical analysis
2. Explains that there was an issue processing the data
3. Suggests they try a different approach or query
4. Maintains a professional and helpful tone

Keep it concise (2-3 sentences) and avoid technical details.
"""
                
                messages = [{"role": "user", "content": prompt}]
                error_message = chat(messages, temperature=0.7, max_tokens=150)
                return QueryResponse(message=error_message.strip())
                
            except Exception:
                # Only use placeholder when LLM is unavailable
                return QueryResponse(
                    message="I'm unable to analyze geographical patent data at the moment. Please try again later or try a different type of query."
                )
    
    def _format_geographical_response(self, country_data: List[Dict], limit: int) -> List[str]:
        """Format geographical analysis response"""
        total_patents = sum(item['count'] for item in country_data)
        
        response_lines = [
            f"**Top {limit} Countries by Applicant Origin:**\n",            f"**Total Patents from Top Countries:** {total_patents:,}\n",
            "**Geographical Distribution:**"
        ]
        
        for i, item in enumerate(country_data, 1):
            country = item['country']
            count = item['count']
            percentage = (count / total_patents * 100) if total_patents > 0 else 0
            response_lines.append(f"  {i}. {country}: {count:,} patents ({percentage:.1f}%)")
        
        return response_lines
