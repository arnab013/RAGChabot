"""
Handler for technology analysis queries
"""
import json
from collections import Counter
from typing import Dict, List, Any
from .base import BaseQueryHandler, ChartGenerator, QueryResponse
from database.models import Patent


class TechnologyAnalysisHandler(BaseQueryHandler):
    """Handler for technology analysis queries"""
    
    def get_query_keywords(self) -> List[str]:
        """Keywords that identify technology analysis queries"""
        return [
            "technology", "tech", "classification", "class", "cpc",
            "ipc", "category", "field", "domain", "sector"
        ]
    
    def handle_query(self, query: str, **kwargs) -> QueryResponse:
        """Handle technology analysis query"""
        try:
            # Use IPC analysis since it's available in the database
            return self._handle_ipc_analysis(query)
            
        except Exception as e:
            error_message = self.generate_error_message(
                query=query,
                error_type="technology_analysis_error",
                technical_error=str(e)
            )
            return QueryResponse(message=error_message)
    
    def _handle_ipc_analysis(self, query: str) -> QueryResponse:
        """Handle IPC classification analysis"""
        # Get IPC data from patents
        patents = self.session.query(Patent.ipc).filter(
            Patent.ipc.isnot(None)
        ).all()
        
        ipc_counter = Counter()
        
        for patent in patents:
            if patent.ipc:
                try:
                    # Parse IPC classifications
                    ipc_data = json.loads(patent.ipc) if isinstance(patent.ipc, str) else patent.ipc
                    if isinstance(ipc_data, list):
                        for ipc in ipc_data:
                            if isinstance(ipc, str) and len(ipc) >= 1:
                                # Extract main section (first letter)
                                main_section = ipc[0].upper()
                                ipc_counter[main_section] += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        
        # Convert to list format
        ipc_distribution = [
            {'category': ipc, 'count': count}
            for ipc, count in ipc_counter.most_common(10)
        ]
        
        # Generate response
        response_lines = self._format_ipc_response(ipc_distribution)
        
        # Generate chart
        if ipc_distribution:
            labels = [item['category'] for item in ipc_distribution]
            values = [item['count'] for item in ipc_distribution]
            chart = ChartGenerator.generate_bar_chart(labels, values, "IPC Classification Distribution")
        else:
            chart = None
          # Generate dynamic insights using LLM
        total_patents = sum(item['count'] for item in ipc_distribution)
        top_ipc = ipc_distribution[0] if ipc_distribution else {'category': 'None', 'count': 0}
        data_summary = f"Technology distribution across {len(ipc_distribution)} IPC sections. Total patents: {total_patents}. Top technology: {top_ipc['category']} with {top_ipc['count']} patents ({(top_ipc['count']/total_patents*100):.1f}% of total)."
        insights = self.generate_dynamic_insights(query, chart, data_summary)
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'ipc_distribution': ipc_distribution},
            insight=insights["insight"],
            takeaway=insights["takeaway"]
        )
    
    def _format_ipc_response(self, ipc_distribution: List[Dict]) -> List[str]:
        """Format IPC analysis response"""
        total_patents = sum(item['count'] for item in ipc_distribution)
        
        response_lines = [
            "**IPC Classification Analysis:**\n",            f"**Total Classified Patents:** {total_patents:,}\n",
            "**Distribution by IPC Section:**"
        ]
        
        # IPC section descriptions
        ipc_sections = {
            'A': 'Human Necessities',
            'B': 'Operations & Transport',
            'C': 'Chemistry & Metallurgy',
            'D': 'Textiles & Paper',
            'E': 'Fixed Constructions',
            'F': 'Mechanical Engineering',
            'G': 'Physics',
            'H': 'Electricity'
        }
        
        for item in ipc_distribution:
            section = item['category']
            count = item['count']
            percentage = (count / total_patents * 100) if total_patents > 0 else 0
            description = ipc_sections.get(section, 'Unknown')
            response_lines.append(f"  • {section} - {description}: {count:,} patents ({percentage:.1f}%)")
        
        return response_lines
