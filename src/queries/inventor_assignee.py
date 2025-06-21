"""
Handler for inventor and assignee analysis queries
"""
import json
from collections import Counter
from typing import Dict, List, Any
from .base import BaseQueryHandler, ChartGenerator, QueryResponse
from database.models import Patent


class InventorAssigneeHandler(BaseQueryHandler):
    """Handler for inventor and assignee analysis queries"""
    
    def get_query_keywords(self) -> List[str]:
        """Keywords that identify inventor/assignee queries"""
        return [
            "inventor", "inventors", "assignee", "assignees", "applicant",
            "company", "organization", "top", "most", "leading", "prolific"
        ]
    
    def handle_query(self, query: str, **kwargs) -> QueryResponse:
        """Handle inventor/assignee analysis query"""
        try:
            query_lower = query.lower()
            
            # Determine specific analysis type
            if any(keyword in query_lower for keyword in ["inventor"]):
                return self._handle_inventor_analysis(query_lower)
            elif any(keyword in query_lower for keyword in ["assignee", "company", "organization", "applicant"]):
                return self._handle_assignee_analysis(query_lower)
            else:
                # Default to combined analysis
                return self._handle_combined_analysis()
                
        except Exception as e:
            error_message = self.generate_error_message(
                query=query,
                error_type="inventor_assignee_error",
                technical_error=str(e)
            )
            return QueryResponse(message=error_message)
    
    def _handle_inventor_analysis(self, query_lower: str) -> QueryResponse:
        """Handle inventor analysis"""
        # Extract limit from query (e.g., "top 10 inventors")
        import re
        limit_match = re.search(r'(?:top|first|leading)\s+(\d+)', query_lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        
        # Get inventor data from inventor_names field (JSON array)
        patents_with_inventors = self.session.query(Patent.inventor_names).filter(
            Patent.inventor_names.isnot(None),
            Patent.inventor_names != ''
        ).all()
        
        inventor_counter = Counter()
        
        for patent in patents_with_inventors:
            if patent.inventor_names:
                try:
                    # Parse JSON array of inventor names
                    inventors = json.loads(patent.inventor_names)
                    if isinstance(inventors, list):
                        for inventor in inventors:
                            if isinstance(inventor, str) and inventor.strip():
                                inventor_counter[inventor.strip()] += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Get top inventors
        top_inventors = inventor_counter.most_common(limit)
        
        # Format data
        inventor_data = [
            {'name': name, 'count': count}
            for name, count in top_inventors
        ]
        
        # Generate response
        response_lines = self._format_inventor_response(inventor_data, limit)
        
        # Generate chart
        if inventor_data:
            labels = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] for item in inventor_data]
            values = [item['count'] for item in inventor_data]
            chart = ChartGenerator.generate_bar_chart(labels, values, f"Top {limit} Inventors by Patent Count")
        else:
            chart = None
            
        # Generate dynamic insights and takeaway
        if inventor_data:
            chart_data_for_insights = {
                'type': 'bar',
                'title': f'Top {limit} Inventors by Patent Count',
                'data': {
                    'labels': labels,
                    'datasets': [{'data': values, 'label': 'Patents'}]
                }
            }
            data_summary = f"Top {limit} inventors by patent count. Data shows {len(inventor_data)} inventors with their patent counts."
            insights = self.generate_dynamic_insights(
                query=query_lower,
                chart_data=chart_data_for_insights,
                data_summary=data_summary
            )
        else:
            insights = {'insight': 'No inventor data available.', 'takeaway': 'Consider expanding the search criteria.'}
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'inventor_stats': inventor_data},
            insight=insights['insight'],
            takeaway=insights['takeaway']
        )
    
    def _handle_assignee_analysis(self, query_lower: str) -> QueryResponse:
        """Handle assignee analysis"""
        # Extract limit from query
        import re
        limit_match = re.search(r'(?:top|first|leading)\s+(\d+)', query_lower)
        # Default to 20 results to improve efficiency and reduce risk of timeout
        limit = int(limit_match.group(1)) if limit_match else 20
        
        # Get assignee data from applicant_names field (JSON array)
        patents_with_applicants = self.session.query(Patent.applicant_names).filter(
            Patent.applicant_names.isnot(None),
            Patent.applicant_names != ''
        ).all()
        
        assignee_counter = Counter()
        
        for patent in patents_with_applicants:
            if patent.applicant_names:
                try:
                    # Parse JSON array of applicant names
                    applicants = json.loads(patent.applicant_names)
                    if isinstance(applicants, list):
                        for applicant in applicants:
                            if isinstance(applicant, str) and applicant.strip():
                                assignee_counter[applicant.strip()] += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Get top assignees - limited to top 20 maximum for efficiency
        actual_limit = min(limit, 20)
        top_assignees = assignee_counter.most_common(actual_limit)
        
        # Format data
        assignee_data = [
            {'name': name, 'count': count}
            for name, count in top_assignees
        ]
        
        # Generate response
        response_lines = self._format_assignee_response(assignee_data, actual_limit)
        
        # Generate chart
        if assignee_data:
            labels = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] for item in assignee_data]
            values = [item['count'] for item in assignee_data]
            chart = ChartGenerator.generate_bar_chart(labels, values, f"Top {actual_limit} Assignees by Patent Count")
        else:
            chart = None
            
        # Generate dynamic insights and takeaway
        if assignee_data:
            chart_data_for_insights = {
                'type': 'bar',
                'title': f'Top {actual_limit} Assignees by Patent Count',
                'data': {
                    'labels': labels,
                    'datasets': [{'data': values, 'label': 'Patents'}]
                }
            }
            data_summary = f"Top {actual_limit} assignees (companies/organizations) by patent count. Data shows {len(assignee_data)} assignees with their patent counts."
            insights = self.generate_dynamic_insights(
                query=query_lower,
                chart_data=chart_data_for_insights,
                data_summary=data_summary
            )
        else:
            insights = {'insight': 'No assignee data available.', 'takeaway': 'Consider expanding the search criteria.'}
        
        return QueryResponse(
            message="\n".join(response_lines),
            chart=chart,
            data={'assignee_stats': assignee_data},
            insight=insights['insight'],
            takeaway=insights['takeaway']
        )
    
    def _handle_combined_analysis(self) -> QueryResponse:
        """Handle combined inventor and assignee analysis"""
        inventor_response = self._handle_inventor_analysis("top 5 inventors")
        assignee_response = self._handle_assignee_analysis("top 5 assignees")
        
        # Combine responses
        combined_message = f"{inventor_response.message}\n\n{assignee_response.message}"
        
        return QueryResponse(
            message=combined_message,
            chart=assignee_response.chart,  # Use assignee chart as primary
            data={
                'inventor_stats': inventor_response.data.get('inventor_stats', []),
                'assignee_stats': assignee_response.data.get('assignee_stats', [])
            },
            insight=assignee_response.insight,  # Use assignee insights for combined view
            takeaway=assignee_response.takeaway
        )
    
    def _format_inventor_response(self, inventor_data: List[Dict], limit: int) -> List[str]:
        """Format inventor analysis response"""
        total_patents = sum(item['count'] for item in inventor_data)
        
        response_lines = [
            f"**Top {limit} Inventors Analysis:**\n",
            f"📊 **Total Patents by Top Inventors:** {total_patents:,}\n",
            "🧑‍💼 **Most Prolific Inventors:**"
        ]
        
        for i, item in enumerate(inventor_data, 1):
            name = item['name']
            count = item['count']
            response_lines.append(f"  {i}. {name}: {count:,} patents")
        
        return response_lines
    
    def _format_assignee_response(self, assignee_data: List[Dict], limit: int) -> List[str]:
        """Format assignee analysis response"""
        total_patents = sum(item['count'] for item in assignee_data)
        
        response_lines = [
            f"**Top {limit} Assignees Analysis:**\n",
            f"📊 **Total Patents by Top Assignees:** {total_patents:,}\n",
            "🏢 **Leading Organizations:**"
        ]
        
        for i, item in enumerate(assignee_data, 1):
            name = item['name']
            count = item['count']
            response_lines.append(f"  {i}. {name}: {count:,} patents")
        
        return response_lines
