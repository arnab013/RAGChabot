"""
Simple working technology analysis handler
"""
import json
from collections import Counter
from typing import Dict, List, Any
from database.models import Patent
from database.config import get_db_session_simple

def _generate_insights_and_takeaways(query, data, chart_type="technology"):
    """Generate insights and takeaways using LLM"""
    try:
        try:
            from src.llm_clients import chat
        except ImportError:
            from llm_clients import chat
        
        # Prepare data summary for LLM
        top_items = data[:5] if len(data) > 5 else data
        
        if chart_type == "technology":
            data_summary = "\n".join([f"- {item['category']}: {item['value']} patents" for item in top_items])
            prompt = f"""Based on the user query "{query}" and the following technology analysis data:

{data_summary}

Generate:
1. Key Insight: One concise, meaningful insight about the technology landscape
2. Key Takeaway: One actionable takeaway or implication for researchers/businesses

Format as JSON:
{{"insight": "Your key insight here", "takeaway": "Your key takeaway here"}}"""
            
        elif chart_type == "inventor":
            data_summary = "\n".join([f"- {item['category']}: {item['value']} patents" for item in top_items])
            prompt = f"""Based on the user query "{query}" and the following inventor analysis data:

{data_summary}

Generate:
1. Key Insight: One concise, meaningful insight about the innovation landscape or prolific inventors
2. Key Takeaway: One actionable takeaway or implication for researchers/businesses

Format as JSON:
{{"insight": "Your key insight here", "takeaway": "Your key takeaway here"}}"""
            
        elif chart_type == "geographic":
            data_summary = "\n".join([f"- {item['category']}: {item['value']} patents" for item in top_items])
            prompt = f"""Based on the user query "{query}" and the following geographical patent analysis data:

{data_summary}

Generate:
1. Key Insight: One concise, meaningful insight about the geographical distribution of innovation
2. Key Takeaway: One actionable takeaway or implication for global innovation trends

Format as JSON:
{{"insight": "Your key insight here", "takeaway": "Your key takeaway here"}}"""
        
        response = chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=200)
        
        # Parse JSON response
        import json
        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            result = json.loads(response_clean.strip())
            return result.get('insight', ''), result.get('takeaway', '')
        except:
            return '', ''
    except:
        return '', ''

def get_top_technology_fields(limit=None):
    """Get top technology fields from the database"""
    session = get_db_session_simple()
    
    try:
        # Get all IPC technologies
        ipc_tech_data = session.query(Patent.ipc_technologies).filter(
            Patent.ipc_technologies.isnot(None),
            Patent.ipc_technologies != ''
        ).all()
        
        # Parse and count technologies
        all_technologies = []
        for tech_str in ipc_tech_data:
            if tech_str[0]:
                try:
                    technologies = json.loads(tech_str[0])
                    all_technologies.extend(technologies)
                except:
                    # Fallback parsing
                    technologies = tech_str[0].replace(';', ',').split(',')
                    all_technologies.extend([t.strip() for t in technologies if t.strip()])
          # Count and get top technologies
        tech_counter = Counter(all_technologies)
        
        # If no limit specified, return all technologies, otherwise limit
        if limit is None:
            top_technologies = tech_counter.most_common()
        else:
            top_technologies = tech_counter.most_common(limit)
        
        # Format as chart data
        chart_data = [
            {"category": tech, "value": count}
            for tech, count in top_technologies
        ]
        
        # Generate response message
        total_count = sum(count for _, count in top_technologies)
        actual_limit = len(top_technologies)
        limit_text = f"All {actual_limit}" if limit is None else f"Top {limit}"
        
        message_lines = [
            f"**{limit_text} Technology Fields by Patent Count**\n",
            f"**Total Patents Analyzed:** {len(ipc_tech_data):,}\n",
            f"**Technology Classifications:** {total_count:,}\n\n"        ]
        for i, (tech, count) in enumerate(top_technologies, 1):
            percentage = (count / total_count * 100) if total_count > 0 else 0
            message_lines.append(f"**{i}. {tech}**")
            message_lines.append(f"   • Patents: {count:,}")
            message_lines.append(f"   • Share: {percentage:.1f}%\n")
        
        # Generate insights and takeaways
        query = "technology fields analysis"  # Default query
        insight, takeaway = _generate_insights_and_takeaways(query, chart_data, "technology")
        
        return {
            'message': '\n'.join(message_lines),
            'chart': {
                'type': 'bar',
                'title': f'{limit_text} Technology Fields',
                'data': chart_data
            },
            'insight': insight,
            'takeaway': takeaway
        }
        
    except Exception as e:
        return {
            'message': f"Error analyzing technology fields: {str(e)}",
            'chart': None
        }
    finally:
        session.close()

def get_top_inventors(limit=None):
    """Get top inventors by patent count"""
    session = get_db_session_simple()
    
    try:
        # Get all inventor names
        inventor_data = session.query(Patent.inventor_names).filter(
            Patent.inventor_names.isnot(None),
            Patent.inventor_names != ''
        ).all()
        
        # Parse and count inventors
        all_inventors = []
        for inventor_str in inventor_data:
            if inventor_str[0]:
                try:
                    inventors = json.loads(inventor_str[0])
                    all_inventors.extend(inventors)
                except:
                    # Fallback parsing
                    inventors = inventor_str[0].replace(';', ',').split(',')
                    all_inventors.extend([i.strip() for i in inventors if i.strip()])
          # Count and get top inventors
        inventor_counter = Counter(all_inventors)
        
        # If no limit specified, return all inventors, otherwise limit
        if limit is None:
            top_inventors = inventor_counter.most_common()
        else:
            top_inventors = inventor_counter.most_common(limit)
        
        # Format as chart data
        chart_data = [
            {"category": inventor, "value": count}
            for inventor, count in top_inventors
        ]
        
        # Generate response message
        total_patents = len(inventor_data)
        actual_limit = len(top_inventors)
        limit_text = f"All {actual_limit}" if limit is None else f"Top {limit}"
        
        message_lines = [
            f"**{limit_text} Inventors by Patent Count**\n",
            f"**Total Patents with Inventor Data:** {total_patents:,}\n",
            f"**Unique Inventors:** {len(inventor_counter):,}\n\n"
        ]
        for i, (inventor, count) in enumerate(top_inventors, 1):
            message_lines.append(f"**{i}. {inventor}**")
            message_lines.append(f"   • Patents: {count:,}\n")
        
        # Generate insights and takeaways
        query = "top inventors analysis"
        insight, takeaway = _generate_insights_and_takeaways(query, chart_data, "inventor")
        
        return {
            'message': '\n'.join(message_lines),
            'chart': {
                'type': 'bar',
                'title': f'{limit_text} Inventors by Patent Count',
                'data': chart_data
            },
            'insight': insight,
            'takeaway': takeaway
        }
        
    except Exception as e:
        return {
            'message': f"Error analyzing inventors: {str(e)}",
            'chart': None
        }
    finally:
        session.close()

def get_patent_counts_by_country(limit=None):
    """Get patent counts by country"""
    session = get_db_session_simple()
    
    try:
        # Get all applicant countries
        country_data = session.query(Patent.applicant_countries).filter(
            Patent.applicant_countries.isnot(None),
            Patent.applicant_countries != ''
        ).all()
        
        # Parse and count countries
        all_countries = []
        for country_str in country_data:
            if country_str[0]:
                try:
                    countries = json.loads(country_str[0])
                    all_countries.extend(countries)
                except:
                    # Fallback parsing
                    countries = country_str[0].replace(';', ',').split(',')
                    all_countries.extend([c.strip() for c in countries if c.strip()])
          # Count and get top countries
        country_counter = Counter(all_countries)
        
        # If no limit specified, return all countries, otherwise limit
        if limit is None:
            top_countries = country_counter.most_common()
        else:
            top_countries = country_counter.most_common(limit)
        
        # Format as chart data
        chart_data = [
            {"category": country, "value": count}
            for country, count in top_countries
        ]
        
        # Generate response message
        total_patents = len(country_data)
        total_count = sum(count for _, count in top_countries)
        actual_limit = len(top_countries)
        limit_text = f"All {actual_limit}" if limit is None else f"Top {limit}"
        
        message_lines = [
            f"**Patent Counts by Country ({limit_text})**\n",
            f"**Total Patents with Country Data:** {total_patents:,}\n",
            f"**Countries Represented:** {len(country_counter):,}\n\n"
        ]
        for i, (country, count) in enumerate(top_countries, 1):
            percentage = (count / total_count * 100) if total_count > 0 else 0
            message_lines.append(f"**{i}. {country}**")
            message_lines.append(f"   • Patents: {count:,}")
            message_lines.append(f"   • Share: {percentage:.1f}%\n")
        
        # Generate insights and takeaways
        query = "patent distribution by country"
        insight, takeaway = _generate_insights_and_takeaways(query, chart_data, "geographic")
        
        return {
            'message': '\n'.join(message_lines),
            'chart': {
                'type': 'bar',
                'title': f'Patent Counts by Country ({limit_text})',
                'data': chart_data
            },
            'insight': insight,
            'takeaway': takeaway
        }
        
    except Exception as e:
        return {
            'message': f"Error analyzing countries: {str(e)}",
            'chart': None
        }
    finally:
        session.close()

if __name__ == "__main__":
    # Test the functions
    print("Testing technology fields...")
    result = get_top_technology_fields(10)
    print(result['message'])
    
    print("\n" + "="*50)
    print("Testing inventors...")
    result = get_top_inventors(10)
    print(result['message'])
    
    print("\n" + "="*50)
    print("Testing countries...")
    result = get_patent_counts_by_country(10)
    print(result['message'])
