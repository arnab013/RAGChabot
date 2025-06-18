"""
Chart template catalog and SQL query templates for patent data visualization.
Following the enhanced roadmap for chart generation.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ChartType(Enum):
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    PIE = "pie"
    DONUT = "donut"
    TREEMAP = "treemap"
    CHOROPLETH = "choropleth"
    GROUPED_BAR = "grouped_bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    NETWORK = "network"
    WORDCLOUD = "wordcloud"

@dataclass
class ChartTemplate:
    """Chart template definition with SQL query and configuration"""
    template_id: str
    description: str
    chart_type: ChartType
    sql_query: str
    parameters: Dict[str, Any]
    data_source: str
    notes: str = ""
    
class ChartTemplateRegistry:
    """Registry of all available chart templates"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all chart templates"""
        
        # 1. Yearly Patent Count - Line/Area Chart
        self.register_template(ChartTemplate(
            template_id="yearly_count",
            description="Publications per Year",
            chart_type=ChartType.LINE,
            sql_query="""
                SELECT 
                    strftime('%Y', publication_date) as year,
                    COUNT(*) as count
                FROM patents 
                WHERE publication_date IS NOT NULL
                    {sdg_filter}
                    {date_filter}
                GROUP BY strftime('%Y', publication_date)
                ORDER BY year
            """,
            parameters={"sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents",
            notes="Time-series showing patent publication trends"
        ))
        
        # 2. SDG Distribution - Bar/Pie Chart
        self.register_template(ChartTemplate(
            template_id="sdg_distribution",
            description="SDG-wise Patent Counts",
            chart_type=ChartType.BAR,
            sql_query="""
                WITH sdg_expanded AS (
                    SELECT 
                        p.publication_number,
                        CAST(json_each.value AS INTEGER) as sdg_number
                    FROM patents p,
                         json_each(p.sdg_number)
                    WHERE p.sdg_number IS NOT NULL 
                        AND p.sdg_number != '[]'
                        {date_filter}
                )
                SELECT 
                    'SDG ' || sdg_number as category,
                    COUNT(*) as count
                FROM sdg_expanded
                WHERE sdg_number BETWEEN 1 AND 17
                GROUP BY sdg_number
                ORDER BY count DESC
                LIMIT {limit}
            """,
            parameters={"limit": 10, "start_year": None, "end_year": None},
            data_source="patents (JSON expanded)",
            notes="Flattened SDG analysis from JSON arrays"
        ))
        
        # 3. Publication Kind Breakdown
        self.register_template(ChartTemplate(
            template_id="kind_breakdown",
            description="Publications by Kind",
            chart_type=ChartType.PIE,
            sql_query="""
                SELECT 
                    COALESCE(publication_kind, 'Unknown') as category,
                    COUNT(*) as count
                FROM patents
                WHERE 1=1 {date_filter} {sdg_filter}
                GROUP BY publication_kind
                ORDER BY count DESC
                LIMIT {limit}
            """,
            parameters={"limit": 8, "sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents",
            notes="Distribution of patent publication types"
        ))
        
        # 4. Geographic Distribution - Applicant Countries
        self.register_template(ChartTemplate(
            template_id="geo_distribution",
            description="Top Applicant Countries",
            chart_type=ChartType.BAR,
            sql_query="""
                WITH countries_expanded AS (
                    SELECT 
                        p.publication_number,
                        TRIM(json_each.value, '"') as country
                    FROM patents p,
                         json_each(p.applicant_countries)
                    WHERE p.applicant_countries IS NOT NULL 
                        AND p.applicant_countries != '[]'
                        {date_filter}
                        {sdg_filter}
                )
                SELECT 
                    country as category,
                    COUNT(*) as count
                FROM countries_expanded
                WHERE country IS NOT NULL AND country != ''
                GROUP BY country
                ORDER BY count DESC
                LIMIT {limit}
            """,
            parameters={"limit": 10, "sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents (applicant_countries JSON)",
            notes="Geographic analysis of patent applicants"
        ))
        
        # 5. IPC Technology Fields Treemap
        self.register_template(ChartTemplate(
            template_id="ipc_treemap",
            description="Technology Fields Treemap",
            chart_type=ChartType.TREEMAP,
            sql_query="""
                WITH ipc_expanded AS (
                    SELECT 
                        p.publication_number,
                        TRIM(json_each.value, '"') as tech_field
                    FROM patents p,
                         json_each(p.ipc_tech_field)
                    WHERE p.ipc_tech_field IS NOT NULL 
                        AND p.ipc_tech_field != '[]'
                        {date_filter}
                        {sdg_filter}
                )
                SELECT 
                    tech_field as category,
                    COUNT(*) as count
                FROM ipc_expanded
                WHERE tech_field IS NOT NULL AND tech_field != ''
                GROUP BY tech_field
                ORDER BY count DESC
                LIMIT {limit}
            """,
            parameters={"limit": 20, "sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents (ipc_tech_field JSON)",
            notes="Hierarchical view of technology classifications"
        ))
        
        # 6. Patent Family Size Distribution
        self.register_template(ChartTemplate(
            template_id="family_sizes",
            description="Patent Family Size Distribution",
            chart_type=ChartType.HISTOGRAM,
            sql_query="""
                WITH family_sizes AS (
                    SELECT 
                        COALESCE(parent_publication_number, publication_number) as family_root,
                        COUNT(*) as family_size
                    FROM patents
                    WHERE 1=1 {date_filter} {sdg_filter}
                    GROUP BY COALESCE(parent_publication_number, publication_number)
                ),
                size_buckets AS (
                    SELECT 
                        CASE 
                            WHEN family_size = 1 THEN '1'
                            WHEN family_size BETWEEN 2 AND 5 THEN '2-5'
                            WHEN family_size BETWEEN 6 AND 10 THEN '6-10'
                            WHEN family_size BETWEEN 11 AND 20 THEN '11-20'
                            ELSE '20+'
                        END as size_bucket,
                        COUNT(*) as count
                    FROM family_sizes
                    GROUP BY size_bucket
                )
                SELECT size_bucket as category, count
                FROM size_buckets
                ORDER BY 
                    CASE size_bucket
                        WHEN '1' THEN 1
                        WHEN '2-5' THEN 2
                        WHEN '6-10' THEN 3
                        WHEN '11-20' THEN 4
                        ELSE 5
                    END
            """,
            parameters={"sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents",
            notes="Distribution of patent family sizes"
        ))
        
        # 7. Chunks per Patent Distribution
        self.register_template(ChartTemplate(
            template_id="chunk_counts",
            description="Chunks per Patent",
            chart_type=ChartType.HISTOGRAM,
            sql_query="""
                WITH chunk_counts AS (
                    SELECT 
                        publication_number,
                        COUNT(*) as chunk_count
                    FROM patent_chunks
                    WHERE 1=1 {date_filter} {sdg_filter}
                    GROUP BY publication_number
                ),
                count_buckets AS (
                    SELECT 
                        CASE 
                            WHEN chunk_count BETWEEN 1 AND 5 THEN '1-5'
                            WHEN chunk_count BETWEEN 6 AND 10 THEN '6-10'
                            WHEN chunk_count BETWEEN 11 AND 20 THEN '11-20'
                            WHEN chunk_count BETWEEN 21 AND 50 THEN '21-50'
                            ELSE '50+'
                        END as count_bucket,
                        COUNT(*) as count
                    FROM chunk_counts
                    GROUP BY count_bucket
                )
                SELECT count_bucket as category, count
                FROM count_buckets
                ORDER BY 
                    CASE count_bucket
                        WHEN '1-5' THEN 1
                        WHEN '6-10' THEN 2
                        WHEN '11-20' THEN 3
                        WHEN '21-50' THEN 4
                        ELSE 5
                    END
            """,
            parameters={"sdg_number": None, "start_year": None, "end_year": None},
            data_source="patent_chunks",
            notes="Distribution of text chunks per patent"
        ))
        
        # 8. Monthly Publication Timeline
        self.register_template(ChartTemplate(
            template_id="monthly_timeline",
            description="Monthly Publication Timeline",
            chart_type=ChartType.AREA,
            sql_query="""
                SELECT 
                    strftime('%Y-%m', publication_date) as month,
                    COUNT(*) as count
                FROM patents 
                WHERE publication_date IS NOT NULL
                    {sdg_filter}
                    {date_filter}
                GROUP BY strftime('%Y-%m', publication_date)
                ORDER BY month
                LIMIT {limit}
            """,
            parameters={"limit": 24, "sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents",
            notes="Monthly trend analysis with area fill"
        ))
        
        # 9. Applicant vs Inventor Countries Comparison
        self.register_template(ChartTemplate(
            template_id="app_vs_inv_countries",
            description="Applicants vs Inventors by Country",
            chart_type=ChartType.GROUPED_BAR,
            sql_query="""
                WITH applicant_countries AS (
                    SELECT 
                        TRIM(json_each.value, '"') as country,
                        COUNT(*) as applicant_count
                    FROM patents p,
                         json_each(p.applicant_countries)
                    WHERE p.applicant_countries IS NOT NULL 
                        AND p.applicant_countries != '[]'
                        {date_filter}
                        {sdg_filter}
                    GROUP BY TRIM(json_each.value, '"')
                ),
                inventor_countries AS (
                    SELECT 
                        TRIM(json_each.value, '"') as country,
                        COUNT(*) as inventor_count
                    FROM patents p,
                         json_each(p.inventor_countries)
                    WHERE p.inventor_countries IS NOT NULL 
                        AND p.inventor_countries != '[]'
                        {date_filter}
                        {sdg_filter}
                    GROUP BY TRIM(json_each.value, '"')
                ),
                top_countries AS (
                    SELECT country 
                    FROM applicant_countries 
                    ORDER BY applicant_count DESC 
                    LIMIT {limit}
                )
                SELECT 
                    tc.country as category,
                    COALESCE(ac.applicant_count, 0) as applicants,
                    COALESCE(ic.inventor_count, 0) as inventors
                FROM top_countries tc
                LEFT JOIN applicant_countries ac ON tc.country = ac.country
                LEFT JOIN inventor_countries ic ON tc.country = ic.country
                ORDER BY COALESCE(ac.applicant_count, 0) DESC
            """,
            parameters={"limit": 10, "sdg_number": None, "start_year": None, "end_year": None},
            data_source="patents (both applicant_countries and inventor_countries JSON)",
            notes="Comparison of applicant vs inventor geographic distribution"
        ))
    
    def register_template(self, template: ChartTemplate):
        """Register a new chart template"""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[ChartTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """List all available template IDs"""
        return list(self.templates.keys())
    
    def get_templates_by_type(self, chart_type: ChartType) -> List[ChartTemplate]:
        """Get all templates of a specific chart type"""
        return [t for t in self.templates.values() if t.chart_type == chart_type]
    
    def build_sql_filters(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Build SQL filter strings from parameters"""
        filters = {
            'sdg_filter': '',
            'date_filter': '',
            'limit': str(parameters.get('limit', 10))
        }
        
        # SDG filter
        if parameters.get('sdg_number'):
            sdg_num = parameters['sdg_number']
            filters['sdg_filter'] = f"AND json_extract(sdg_number, '$') LIKE '%{sdg_num}%'"
        
        # Date filters
        date_conditions = []
        if parameters.get('start_year'):
            date_conditions.append(f"strftime('%Y', publication_date) >= '{parameters['start_year']}'")
        if parameters.get('end_year'):
            date_conditions.append(f"strftime('%Y', publication_date) <= '{parameters['end_year']}'")
        
        if date_conditions:
            filters['date_filter'] = "AND " + " AND ".join(date_conditions)
        
        return filters
    
    def get_template_descriptions(self) -> Dict[str, str]:
        """Get template ID to description mapping for LLM context"""
        return {tid: template.description for tid, template in self.templates.items()}

# Global registry instance
chart_registry = ChartTemplateRegistry()
