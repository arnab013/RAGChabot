"""
Handler for SDG (Sustainable Development Goals) distribution queries
"""
import json
from collections import Counter
from typing import Dict, List, Any
from .base import BaseQueryHandler, ChartGenerator, QueryResponse
from database.models import Patent


class SDGDistributionHandler(BaseQueryHandler):
    """Handler for SDG distribution queries"""
    
    def get_query_keywords(self) -> List[str]:
        """Keywords that identify SDG distribution queries"""
        return [
            "sdg", "sdgs", "sustainable development", "sustainable development goals",
            "sdg distribution", "sdg breakdown", "categories", "goals"
        ]
    
    def handle_query(self, query: str, **kwargs) -> QueryResponse:
        """Handle SDG distribution query"""
        try:
            # Get SDG data
            sdg_data = self._get_sdg_distribution()
            
            if not sdg_data:
                return QueryResponse(message="Sorry, I couldn't retrieve the SDG distribution data.")
            
            # Generate response
            response_lines = self._format_sdg_response(sdg_data)
            
            # Generate chart
            chart = self._generate_sdg_chart(sdg_data)
            
            # Generate dynamic insights using LLM
            total_patents = sum(item['count'] for item in sdg_data)
            top_sdg = sdg_data[0] if sdg_data else {'sdg': 'None', 'count': 0}
            data_summary = f"SDG distribution across {len(sdg_data)} categories. Total patents: {total_patents}. Top SDG: {top_sdg['sdg']} with {top_sdg['count']} patents ({(top_sdg['count']/total_patents*100):.1f}% of total)."
            insights = self.generate_dynamic_insights(query, chart, data_summary)
            
            return QueryResponse(
                message="\n".join(response_lines),
                chart=chart,                data={'sdg_distribution': sdg_data},
                insight=insights["insight"],
                takeaway=insights["takeaway"]
            )
            
        except Exception as e:
            error_message = self.generate_error_message(
                query=query,
                error_type="sdg_distribution_error",
                technical_error=str(e)
            )
            return QueryResponse(message=error_message)
    
    def _get_sdg_distribution(self) -> List[Dict[str, Any]]:
        """Get SDG distribution from database"""
        patents = self.session.query(Patent.sdg_number).filter(
            Patent.sdg_number.isnot(None)
        ).all()
        
        sdg_counter = Counter()
        
        for patent in patents:
            if patent.sdg_number:
                try:
                    sdg_list = json.loads(patent.sdg_number)
                    if isinstance(sdg_list, list):
                        for sdg in sdg_list:
                            if isinstance(sdg, (int, str)) and str(sdg).isdigit():
                                sdg_counter[int(sdg)] += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Convert to list format
        return [
            {'sdg': sdg, 'count': count}
            for sdg, count in sdg_counter.most_common(17)  # Max 17 SDGs
        ]
    
    def _format_sdg_response(self, sdg_data: List[Dict]) -> List[str]:
        """Format SDG distribution response"""
        total_patents = sum(item['count'] for item in sdg_data)
        
        response_lines = [
            "**SDG Distribution:**\n",
            f"📊 **Total SDG-Classified Patents:** {total_patents:,}\n",
            "📈 **Distribution by SDG:**"
        ]
        
        for item in sdg_data:
            sdg = item['sdg']
            count = item['count']
            percentage = (count / total_patents * 100) if total_patents > 0 else 0
            response_lines.append(f"  • SDG {sdg}: {count:,} patents ({percentage:.1f}%)")
        
        return response_lines
    
    def _generate_sdg_chart(self, sdg_data: List[Dict]) -> Dict[str, Any]:
        """Generate chart for SDG distribution"""
        if not sdg_data:
            return None
        
        labels = [f"SDG {item['sdg']}" for item in sdg_data]
        values = [item['count'] for item in sdg_data]
        
        # Use pie chart for SDG distribution
        return {
            'type': 'pie',
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': values,
                    'backgroundColor': [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
                        '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56',
                        '#9966FF', '#FF9F40', '#4BC0C0', '#C9CBCF', '#FF6384'
                    ]
                }]
            },
            'title': 'SDG Distribution'
        }
