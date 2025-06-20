"""
Statistical queries for patent data analysis and visualization.
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

def determine_chart_type(data, x_column_name, query_context=""):
    """
    Determine the appropriate chart type based on data characteristics and query context.
    Returns chart type and formatted data for visualization.
    
    Chart types supported:
    - line: Trends over time (year, month)
    - bar: Categorical comparisons (vertical only)
    - pie: Proportional data (≤8 slices)
    - treemap: Hierarchical/dense categorical data
    - area: Trends with filled area emphasis
    - stacked_bar: Category within category
    """
    if not data or len(data) == 0:
        return None, None
    
    query_lower = query_context.lower()
    x_column_lower = x_column_name.lower()
    
    # Check for specific chart type indicators in query
    if any(keyword in query_lower for keyword in ['share', 'percentage', 'proportion', 'distribution']) and len(data) <= 8:
        return 'pie', format_chart_data(data, 'pie', x_column_name)
    
    # Check for temporal data (line chart)
    temporal_keywords = ['year', 'date', 'month', 'time', 'trend']
    is_temporal = any(keyword in x_column_lower for keyword in temporal_keywords)
    
    # Check for cumulative or area indicators
    if any(keyword in query_lower for keyword in ['cumulative', 'growth', 'accumulation', 'volume']):
        return 'area', format_chart_data(data, 'area', x_column_name)
    
    # Check for hierarchical data indicators (treemap)
    if any(keyword in query_lower for keyword in ['visualize', 'hierarchy', 'breakdown']) and len(data) > 10:
        return 'treemap', format_chart_data(data, 'treemap', x_column_name)
    
    # Time-based data defaults to line chart
    if is_temporal:
        return 'line', format_chart_data(data, 'line', x_column_name)
    
    # Categorical data with small count - pie chart
    elif len(data) <= 8:
        return 'pie', format_chart_data(data, 'pie', x_column_name)
    
    # Large categorical data - bar chart
    else:
        return 'bar', format_chart_data(data, 'bar', x_column_name)

def format_chart_data(data, chart_type, x_column_name=""):
    """Format data for frontend chart consumption based on chart type."""
    
    # Color palettes for different chart types
    colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
        '#FF9F40', '#E7E9ED', '#71B37C', '#F7464A', '#46BFBD',
        '#FDB45C', '#949FB1', '#4D5360', '#AC64AD', '#8CC152'
    ]
    
    if chart_type == 'pie':
        return {
            'labels': [str(item['category']) for item in data],
            'datasets': [{
                'data': [item['value'] for item in data],
                'backgroundColor': colors[:len(data)],
                'borderWidth': 2,
                'borderColor': '#ffffff'
            }]
        }
    
    elif chart_type == 'line':
        return {
            'labels': [str(item['category']) for item in data],
            'datasets': [{
                'label': 'Patents',
                'data': [item['value'] for item in data],
                'borderColor': '#36A2EB',
                'backgroundColor': 'rgba(54, 162, 235, 0.1)',
                'fill': False,
                'tension': 0.4,
                'pointBackgroundColor': '#36A2EB',
                'pointBorderColor': '#ffffff',
                'pointBorderWidth': 2
            }]
        }
    
    elif chart_type == 'area':
        return {
            'labels': [str(item['category']) for item in data],
            'datasets': [{
                'label': 'Patents',
                'data': [item['value'] for item in data],
                'borderColor': '#36A2EB',
                'backgroundColor': 'rgba(54, 162, 235, 0.3)',
                'fill': True,
                'tension': 0.4,
                'pointBackgroundColor': '#36A2EB',
                'pointBorderColor': '#ffffff',
                'pointBorderWidth': 2
            }]
        }
    
    elif chart_type == 'bar':
        return {
            'labels': [str(item['category']) for item in data],
            'datasets': [{
                'label': 'Patents',
                'data': [item['value'] for item in data],
                'backgroundColor': colors[1],  # Blue
                'borderColor': colors[1],
                'borderWidth': 1,
                'borderRadius': 4,
                'borderSkipped': False
            }]
        }
    
    elif chart_type == 'treemap':
        # Treemap data format: array of {name, value}
        return {
            'datasets': [{
                'label': 'Patents',
                'data': [
                    {
                        'name': str(item['category']),
                        'value': item['value'],
                        'backgroundColor': colors[i % len(colors)]
                    }
                    for i, item in enumerate(data)
                ],
                'backgroundColor': colors[:len(data)]
            }]
        }
    
    elif chart_type == 'stacked_bar':
        # For stacked bar, we need multi-dimensional data
        # This will be handled in specific methods that support stacking
        return {
            'labels': [str(item['category']) for item in data],
            'datasets': [{
                'label': 'Patents',
                'data': [item['value'] for item in data],
                'backgroundColor': colors[0],
                'borderColor': colors[0],
                'borderWidth': 1
            }]
        }
    
    else:
        # Default to bar chart format
        return format_chart_data(data, 'bar', x_column_name)

def should_generate_chart(query_result, query_description=""):
    """
    Determine if a chart should be generated based on query criteria.
    Returns True only if:
    - Result has exactly 2 columns (categorical + numeric)
    - Data is aggregated/grouped
    - Not text-heavy content
    """
    if not query_result or len(query_result) == 0:
        return False
    
    # Check if result has exactly 2 columns
    # Handle both tuples/lists and SQLAlchemy Row objects
    first_row = query_result[0]
    try:
        # Check if we can access two elements (works for tuples, lists, and Row objects)
        if len(first_row) == 2:
            # Check if second column is numeric
            try:
                numeric_values = [float(row[1]) for row in query_result]
                # Must have at least 2 data points for meaningful visualization
                return len(query_result) >= 2
            except (ValueError, TypeError):
                return False
    except (TypeError, AttributeError):
        return False
    
    return False

class PatentStatistics:
    """Handles statistical queries and data visualization for patent data."""
    
    def __init__(self):
        self.session = get_db_session_simple()
        
    def __del__(self):
        if self.session:
            self.session.close()
    
    def get_basic_stats(self):
        """Get basic statistics about the patent database."""
        try:
            total_patents = self.session.query(Patent).count()
            total_chunks = self.session.query(PatentChunk).count()
            
            # Date range
            date_range = self.session.query(
                func.min(Patent.publication_date),
                func.max(Patent.publication_date)
            ).first()
            
            # Countries with most patents - This generates chart data
            country_stats = self.session.query(
                Patent.publication_country,
                func.count(Patent.publication_number)
            ).group_by(Patent.publication_country).order_by(
                func.count(Patent.publication_number).desc()
            ).limit(10).all()
            
            stats = {
                'total_patents': total_patents,
                'total_chunks': total_chunks,
                'date_range': {
                    'earliest': date_range[0].isoformat() if date_range[0] else None,
                    'latest': date_range[1].isoformat() if date_range[1] else None
                },                'top_countries': [
                    {'country': country, 'count': count}
                    for country, count in country_stats
                ]
            }
            
            # Check if we should generate a chart for country distribution
            if should_generate_chart(country_stats):
                chart_data = [{'category': country, 'value': count} for country, count in country_stats]
                chart_type, formatted_data = determine_chart_type(chart_data, 'country', 'countries distribution')
                stats['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': 'Patents by Country'
                }
            
            logger.info(f"Generated basic stats: {total_patents} patents, {total_chunks} chunks")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting basic stats: {e}")
            return None

    def get_publication_trends(self, years=5, months=None):
        """Get publication trends over time."""
        try:
            # Get publications by year
            yearly_stats = self.session.query(
                extract('year', Patent.publication_date).label('year'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date >= datetime.now() - timedelta(days=years*365)
            ).group_by(
                extract('year', Patent.publication_date)
            ).order_by('year').all()
            
            # Get monthly stats - adjust time range based on request
            months_back = months if months else 12
            monthly_stats = self.session.query(
                extract('year', Patent.publication_date).label('year'),
                extract('month', Patent.publication_date).label('month'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date >= datetime.now() - timedelta(days=months_back*30)
            ).group_by(
                extract('year', Patent.publication_date),
                extract('month', Patent.publication_date)
            ).order_by('year', 'month').all()
            
            trends = {
                'yearly': [
                    {'year': int(year), 'count': count}
                    for year, count in yearly_stats
                ],
                'monthly': [
                    {'year': int(year), 'month': int(month), 'count': count}
                    for year, month, count in monthly_stats
                ]
            }
              # Generate chart based on request type
            if months:
                # Generate monthly chart
                # Format monthly data for chart first (e.g., "2025-03")
                chart_data = [
                    {
                        'category': f"{int(year)}-{int(month):02d}", 
                        'value': count
                    } 
                    for year, month, count in monthly_stats
                ]
                
                # Create 2-column data for should_generate_chart validation
                chart_tuples = [(item['category'], item['value']) for item in chart_data]
                
                if should_generate_chart(chart_tuples):
                    chart_type, formatted_data = determine_chart_type(chart_data, 'month', 'publication trends by month')
                    trends['chart'] = {
                        'type': chart_type,
                        'data': formatted_data,
                        'title': f'Patent Publications by Month (Last {months} Months)'
                    }
            else:
                # Generate yearly chart (default behavior)
                if should_generate_chart(yearly_stats):
                    chart_data = [{'category': int(year), 'value': count} for year, count in yearly_stats]
                    chart_type, formatted_data = determine_chart_type(chart_data, 'year', 'publication trends by year')
                    trends['chart'] = {
                        'type': chart_type,
                        'data': formatted_data,
                        'title': f'Patent Publications by Year (Last {years} Years)'
                    }
            
            logger.info(f"Generated publication trends: {len(trends['yearly'])} years, {len(trends['monthly'])} months")
            return trends
            
        except Exception as e:
            logger.error(f"Error getting publication trends: {e}")
            return None
    
    def get_sdg_distribution(self):
        """Get distribution of SDG (Sustainable Development Goals) categories."""
        try:
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
            
            # Convert to list format for frontend
            sdg_distribution = [
                {'sdg': sdg, 'count': count}
                for sdg, count in sdg_counter.most_common(17)  # Max 17 SDGs
            ]
            
            # Prepare data for chart generation
            chart_data_raw = [(sdg, count) for sdg, count in sdg_counter.most_common(10)]  # Top 10 for chart
            
            result = {'distribution': sdg_distribution}
              # Generate chart for SDG distribution
            if should_generate_chart(chart_data_raw):
                chart_data = [{'category': f'SDG {sdg}', 'value': count} for sdg, count in chart_data_raw]
                chart_type, formatted_data = determine_chart_type(chart_data, 'sdg', 'SDG distribution share')
                result['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': 'Top 10 SDG Distribution'
                }
            
            logger.info(f"Generated SDG distribution: {len(sdg_distribution)} categories")
            return result
            
        except Exception as e:
            logger.error(f"Error getting SDG distribution: {e}")
            return None
    
    def get_technology_fields(self):
        """Get distribution of technology fields from IPC classifications."""
        try:
            patents = self.session.query(Patent.ipc_tech_field).filter(
                Patent.ipc_tech_field.isnot(None)
            ).all()
            
            tech_counter = Counter()
            for patent in patents:
                if patent.ipc_tech_field:
                    try:
                        tech_list = json.loads(patent.ipc_tech_field)
                        if isinstance(tech_list, list):
                            for tech in tech_list:
                                if isinstance(tech, str) and tech.strip():
                                    tech_counter[tech.strip()] += 1
                    except json.JSONDecodeError:
                        continue
            
            # Get top 15 technology fields
            tech_distribution = [
                {'field': field, 'count': count}
                for field, count in tech_counter.most_common(15)
            ]
            
            # Prepare data for chart generation
            chart_data_raw = [(field, count) for field, count in tech_counter.most_common(10)]  # Top 10 for chart
            
            result = {'distribution': tech_distribution}
              # Generate chart for technology fields
            if should_generate_chart(chart_data_raw):
                chart_data = [{'category': field, 'value': count} for field, count in chart_data_raw]
                chart_type, formatted_data = determine_chart_type(chart_data, 'field', 'technology fields distribution visualize')
                result['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': 'Top 10 Technology Fields'
                }
            
            logger.info(f"Generated technology fields distribution: {len(tech_distribution)} fields")
            return result
            
        except Exception as e:
            logger.error(f"Error getting technology fields: {e}")
            return None
    
    def get_applicant_analysis(self, top_n=20):
        """Get analysis of top patent applicants."""
        try:
            patents = self.session.query(
                Patent.applicant_names,
                Patent.applicant_countries,
                Patent.applicant_count
            ).filter(
                Patent.applicant_names.isnot(None)
            ).all()
            
            applicant_counter = Counter()
            country_applicant_map = defaultdict(set)
            
            for patent in patents:
                if patent.applicant_names:
                    try:
                        applicant_list = json.loads(patent.applicant_names)
                        country_list = json.loads(patent.applicant_countries) if patent.applicant_countries else []
                        
                        if isinstance(applicant_list, list):
                            for i, applicant in enumerate(applicant_list):
                                if isinstance(applicant, str) and applicant.strip():
                                    clean_applicant = applicant.strip()
                                    applicant_counter[clean_applicant] += 1
                                    
                                    # Map applicant to country if available
                                    if i < len(country_list) and country_list[i]:
                                        country_applicant_map[country_list[i]].add(clean_applicant)
                                        
                    except json.JSONDecodeError:
                        continue
            
            # Top applicants
            top_applicants = [
                {'applicant': applicant, 'count': count}
                for applicant, count in applicant_counter.most_common(top_n)
            ]
            
            # Applicants by country
            country_stats = {}
            for country, applicants in country_applicant_map.items():
                total_patents = sum(applicant_counter[applicant] for applicant in applicants)
                country_stats[country] = {
                    'unique_applicants': len(applicants),
                    'total_patents': total_patents
                }
            
            analysis = {
                'top_applicants': top_applicants,
                'country_breakdown': [
                    {'country': country, **stats}
                    for country, stats in sorted(
                        country_stats.items(),
                        key=lambda x: x[1]['total_patents'],
                        reverse=True
                    )[:15]
                ]
            }
              # Generate chart for top applicants
            chart_data_raw = [(app['applicant'], app['count']) for app in top_applicants[:10]]  # Top 10 for chart
            if should_generate_chart(chart_data_raw):
                chart_data = [{'category': applicant, 'value': count} for applicant, count in chart_data_raw]
                chart_type, formatted_data = determine_chart_type(chart_data, 'applicant', 'top patent applicants')
                analysis['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': 'Top 10 Patent Applicants'
                }
            
            logger.info(f"Generated applicant analysis: {len(top_applicants)} top applicants")
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting applicant analysis: {e}")
            return None
        
    def get_sdg_by_country_breakdown(self, top_countries=5, top_sdgs=5):
        """Get SDG distribution within each top country (stacked bar chart)."""
        try:
            # Get patents with both country and SDG data
            patents = self.session.query(
                Patent.publication_country,
                Patent.sdg_number
            ).filter(
                Patent.publication_country.isnot(None),
                Patent.sdg_number.isnot(None)
            ).all()
            
            country_sdg_map = defaultdict(lambda: defaultdict(int))
            
            for patent in patents:
                if patent.publication_country and patent.sdg_number:
                    try:
                        sdg_list = json.loads(patent.sdg_number)
                        if isinstance(sdg_list, list):
                            for sdg in sdg_list:
                                if isinstance(sdg, (int, str)) and str(sdg).isdigit():
                                    country_sdg_map[patent.publication_country][int(sdg)] += 1
                    except json.JSONDecodeError:
                        continue
            
            # Get top countries by total patents
            country_totals = {country: sum(sdgs.values()) for country, sdgs in country_sdg_map.items()}
            top_country_names = sorted(country_totals.keys(), key=lambda x: country_totals[x], reverse=True)[:top_countries]
            
            # Get top SDGs globally
            global_sdg_counts = defaultdict(int)
            for sdgs in country_sdg_map.values():
                for sdg, count in sdgs.items():
                    global_sdg_counts[sdg] += count
            top_sdg_numbers = sorted(global_sdg_counts.keys(), key=lambda x: global_sdg_counts[x], reverse=True)[:top_sdgs]
            
            # Build stacked bar data
            stacked_data = []
            for country in top_country_names:
                country_data = {'country': country}
                for sdg in top_sdg_numbers:
                    country_data[f'SDG{sdg}'] = country_sdg_map[country].get(sdg, 0)
                stacked_data.append(country_data)
            
            result = {
                'breakdown': stacked_data,
                'top_countries': top_country_names,
                'top_sdgs': top_sdg_numbers
            }
            
            # Generate stacked bar chart
            if stacked_data:
                result['chart'] = {
                    'type': 'stacked_bar',
                    'data': self._format_stacked_bar_data(stacked_data, top_sdg_numbers),
                    'title': f'SDG Distribution within Top {top_countries} Countries'
                }
            
            logger.info(f"Generated SDG by country breakdown: {len(stacked_data)} countries, {len(top_sdg_numbers)} SDGs")
            return result
            
        except Exception as e:
            logger.error(f"Error getting SDG by country breakdown: {e}")
            return None
    
    def get_cumulative_patent_growth(self, years=10):
        """Get cumulative patent growth over time (area chart)."""
        try:
            # Get patents by year
            yearly_stats = self.session.query(
                extract('year', Patent.publication_date).label('year'),
                func.count(Patent.publication_number).label('count')
            ).filter(
                Patent.publication_date >= datetime.now() - timedelta(days=years*365)
            ).group_by(
                extract('year', Patent.publication_date)
            ).order_by('year').all()
            
            # Calculate cumulative counts
            cumulative_data = []
            running_total = 0
            for year, count in yearly_stats:
                running_total += count
                cumulative_data.append({
                    'year': int(year),
                    'annual_count': count,
                    'cumulative_count': running_total
                })
            
            result = {
                'growth_data': cumulative_data,
                'total_growth': running_total if cumulative_data else 0
            }
            
            # Generate area chart for cumulative growth
            if cumulative_data:
                chart_data = [{'category': item['year'], 'value': item['cumulative_count']} for item in cumulative_data]
                chart_type, formatted_data = determine_chart_type(chart_data, 'year', 'cumulative growth over time')
                result['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': f'Cumulative Patent Growth (Last {years} Years)'
                }
            
            logger.info(f"Generated cumulative growth data: {len(cumulative_data)} years")
            return result
            
        except Exception as e:
            logger.error(f"Error getting cumulative patent growth: {e}")
            return None
    
    def get_technology_field_treemap(self, min_patents=10):
        """Get technology field distribution as treemap for dense visualization."""
        try:
            patents = self.session.query(Patent.ipc_tech_field).filter(
                Patent.ipc_tech_field.isnot(None)
            ).all()
            
            tech_counter = Counter()
            for patent in patents:
                if patent.ipc_tech_field:
                    try:
                        tech_list = json.loads(patent.ipc_tech_field)
                        if isinstance(tech_list, list):
                            for tech in tech_list:
                                if isinstance(tech, str) and tech.strip():
                                    tech_counter[tech.strip()] += 1
                    except json.JSONDecodeError:
                        continue
            
            # Filter out fields with too few patents
            filtered_techs = [(field, count) for field, count in tech_counter.items() if count >= min_patents]
            filtered_techs.sort(key=lambda x: x[1], reverse=True)
            
            result = {
                'technology_fields': [
                    {'field': field, 'count': count}
                    for field, count in filtered_techs
                ]
            }
              # Generate treemap
            if filtered_techs:
                chart_data = [{'category': field, 'value': count} for field, count in filtered_techs]
                chart_type, formatted_data = determine_chart_type(chart_data, 'field', 'visualize technology field hierarchy')
                result['chart'] = {
                    'type': chart_type,
                    'data': formatted_data,
                    'title': f'Technology Fields Distribution (≥{min_patents} patents)'
                }
            
            logger.info(f"Generated technology treemap: {len(filtered_techs)} fields")
            return result
            
        except Exception as e:
            logger.error(f"Error getting technology field treemap: {e}")
            return None

    def get_sdg_trends_over_time(self, years=10, use_available_data=True):
        """Get SDG distribution trends over the specified number of years."""
        try:
            from datetime import datetime, timedelta
            
            if use_available_data:
                # Check data distribution to make an intelligent choice
                recent_cutoff = datetime.now().year - years
                
                # Count patents in recent vs historical periods
                recent_count = self.session.query(Patent).filter(
                    Patent.sdg_number.isnot(None),
                    Patent.publication_date.isnot(None),
                    func.extract('year', Patent.publication_date) >= recent_cutoff
                ).count()
                
                historical_count = self.session.query(Patent).filter(
                    Patent.sdg_number.isnot(None),
                    Patent.publication_date.isnot(None),
                    func.extract('year', Patent.publication_date) < recent_cutoff
                ).count()
                
                # If historical data has significantly more patents, use historical range
                if historical_count > recent_count * 2:
                    # Use all available data
                    patents = self.session.query(
                        Patent.sdg_number, 
                        Patent.publication_date
                    ).filter(
                        Patent.sdg_number.isnot(None),
                        Patent.publication_date.isnot(None)
                    ).all()
                else:
                    # Use the requested recent range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=years * 365)
                    
                    patents = self.session.query(
                        Patent.sdg_number, 
                        Patent.publication_date
                    ).filter(
                        Patent.sdg_number.isnot(None),
                        Patent.publication_date.isnot(None),
                        Patent.publication_date >= start_date.strftime('%Y-%m-%d'),
                        Patent.publication_date <= end_date.strftime('%Y-%m-%d')
                    ).all()
            else:
                # Use exact date range as specified
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years * 365)
                
                patents = self.session.query(
                    Patent.sdg_number, 
                    Patent.publication_date
                ).filter(
                    Patent.sdg_number.isnot(None),
                    Patent.publication_date.isnot(None),
                    Patent.publication_date >= start_date.strftime('%Y-%m-%d'),
                    Patent.publication_date <= end_date.strftime('%Y-%m-%d')
                ).all()
            
            # Create a nested structure: year -> SDG -> count
            yearly_sdg_data = defaultdict(lambda: defaultdict(int))
            sdg_totals = defaultdict(int)
            
            for patent in patents:
                if patent.sdg_number and patent.publication_date:
                    try:
                        # Parse publication date
                        pub_date = datetime.strptime(patent.publication_date, '%Y-%m-%d')
                        year = pub_date.year
                        
                        # Parse SDG numbers
                        sdg_list = json.loads(patent.sdg_number)
                        if isinstance(sdg_list, list):
                            for sdg in sdg_list:
                                if isinstance(sdg, (int, str)) and str(sdg).isdigit():
                                    sdg_num = int(sdg)
                                    yearly_sdg_data[year][sdg_num] += 1
                                    sdg_totals[sdg_num] += 1
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
            
            # Get top 5 SDGs by total count for cleaner visualization
            top_sdgs = sorted(sdg_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            top_sdg_numbers = [sdg for sdg, _ in top_sdgs]
            
            # Prepare chart data - each SDG as a separate line
            chart_data = []
            years_in_range = sorted(yearly_sdg_data.keys())
            
            for year in years_in_range:
                year_data = {'year': year}
                for sdg_num in top_sdg_numbers:
                    year_data[f'SDG {sdg_num}'] = yearly_sdg_data[year].get(sdg_num, 0)
                chart_data.append(year_data)
            
            # Create summary statistics
            total_patents_with_sdg = sum(sdg_totals.values())
            result = {
                'yearly_trends': chart_data,
                'top_sdgs': [{'sdg': sdg, 'total_count': count} for sdg, count in top_sdgs],
                'total_patents': total_patents_with_sdg,
                'years_covered': f"{min(years_in_range) if years_in_range else 'N/A'} - {max(years_in_range) if years_in_range else 'N/A'}",
                'chart': {
                    'type': 'line',
                    'data': {
                        'labels': [str(year) for year in years_in_range],
                        'datasets': [
                            {
                                'label': f'SDG {sdg_num}',
                                'data': [yearly_sdg_data[year].get(sdg_num, 0) for year in years_in_range],
                                'borderColor': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'][i % 5],
                                'backgroundColor': f"rgba({['255,99,132', '54,162,235', '255,206,86', '75,192,192', '153,102,255'][i % 5]}, 0.1)",
                                'fill': False,
                                'tension': 0.4
                            } for i, sdg_num in enumerate(top_sdg_numbers)
                        ]
                    },                    'title': f'SDG Patent Trends Over Last {years} Years'
                }
            }
            
            logger.info(f"Generated SDG trends for {len(years_in_range)} years, {len(top_sdg_numbers)} top SDGs")
            return result
            
        except Exception as e:
            logger.error(f"Error getting SDG trends over time: {e}")
            logger.error(f"Exception in get_sdg_trends_over_time: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_stacked_bar_data(self, stacked_data, sdg_numbers):
        """Format data for stacked bar chart."""
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
            '#FF9F40', '#E7E9ED', '#71B37C', '#F7464A', '#46BFBD'
        ]
        
        return {
            'labels': [item['country'] for item in stacked_data],
            'datasets': [
                {
                    'label': f'SDG {sdg}',
                    'data': [item.get(f'SDG{sdg}', 0) for item in stacked_data],
                    'backgroundColor': colors[i % len(colors)],
                    'borderColor': colors[i % len(colors)],
                    'borderWidth': 1
                }
                for i, sdg in enumerate(sdg_numbers)
            ]
        }
    
    def close(self):
        """Close database session."""
        if self.session:
            self.session.close()
    
    def get_patent_coverage(self, publication_number):
        """
        Get coverage information for a specific patent:
        - Active countries (contracting states)
        - Extended countries (extension states)
        - Time since publication/activation
        
        Args:
            publication_number: The publication number of the patent
            
        Returns:
            Dict with active_count, extension_count, and active_duration information
        """
        try:
            # Query the patent
            patent = self.session.query(Patent).filter(Patent.publication_number == publication_number).first()
            
            if not patent:
                logger.warning(f"Patent {publication_number} not found")
                return None
            
            # Get the counts from JSON arrays
            active_countries = []
            extension_countries = []
            
            if patent.designated_states_contracting:
                try:
                    active_countries = json.loads(patent.designated_states_contracting)
                except json.JSONDecodeError:
                    active_countries = []
            
            if patent.designated_states_extension:
                try:
                    extension_countries = json.loads(patent.designated_states_extension)
                except json.JSONDecodeError:
                    extension_countries = []
            
            # Calculate time since publication
            days_active = 0
            years_active = 0
            months_active = 0
            active_duration_text = "Unknown"
            
            if patent.publication_date:
                days_active = (datetime.now().date() - patent.publication_date).days
                years_active = days_active // 365
                remaining_days = days_active % 365
                months_active = remaining_days // 30
                
                # Format the duration text
                if years_active > 0:
                    active_duration_text = f"{years_active} {'year' if years_active == 1 else 'years'}"
                    if months_active > 0:
                        active_duration_text += f", {months_active} {'month' if months_active == 1 else 'months'}"
                elif months_active > 0:
                    active_duration_text = f"{months_active} {'month' if months_active == 1 else 'months'}"
                else:
                    active_duration_text = f"{days_active} {'day' if days_active == 1 else 'days'}"
            
            # Create the response with detailed information
            response = {
                'active_count': len(active_countries),
                'extension_count': len(extension_countries),
                'active_countries': active_countries,
                'extension_countries': extension_countries,
                'active_duration': {
                    'days': days_active,
                    'years': years_active,
                    'months': months_active,
                    'text': active_duration_text
                },
                'publication_date': patent.publication_date.isoformat() if patent.publication_date else None,
                'publication_country': patent.publication_country,
                'publication_kind': patent.publication_kind,
                'title': patent.title_en
            }
            
            # Generate chart data for donut/gauge visualization
            active_chart_data = [
                {'category': 'Active', 'value': len(active_countries)},
                {'category': 'Potential', 'value': max(0, 38 - len(active_countries))}  # Assuming EPC has 38 member states
            ]
            
            extension_chart_data = [
                {'category': 'Extended', 'value': len(extension_countries)},
                {'category': 'Non-Extended', 'value': max(0, len(active_countries) - len(extension_countries))}
            ]
            
            duration_chart_data = [
                {'category': 'Elapsed', 'value': years_active},
                {'category': 'Remaining', 'value': max(0, 20 - years_active)}  # Assuming 20-year patent term
            ]
            
            # Determine chart types and format data
            active_chart_type, active_chart_data = determine_chart_type(
                active_chart_data, 'country_count', 'active countries'
            )
            
            extension_chart_type, extension_chart_data = determine_chart_type(
                extension_chart_data, 'extension_count', 'extension countries'
            )
            
            duration_chart_type, duration_chart_data = determine_chart_type(
                duration_chart_data, 'years_active', 'patent duration'
            )
            
            # Create chart configuration
            response['chart'] = {
                'type': 'patent_coverage',
                'data': {
                    'active': {
                        'type': active_chart_type or 'pie',
                        'data': active_chart_data,
                        'title': 'Active Countries'
                    },
                    'extension': {
                        'type': extension_chart_type or 'pie',
                        'data': extension_chart_data,
                        'title': 'Extended Countries'
                    },
                    'duration': {
                        'type': duration_chart_type or 'gauge',
                        'data': duration_chart_data,
                        'title': 'Patent Duration'
                    }
                }
            }
            
            logger.info(f"Generated patent coverage for {publication_number}: {len(active_countries)} active, {len(extension_countries)} extended")
            return response
            
        except Exception as e:
            logger.error(f"Error getting patent coverage for {publication_number}: {e}")
            return None
