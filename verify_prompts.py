#!/usr/bin/env python3
"""
Verify that specific prompts can be answered correctly by checking database schema and testing queries.
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from database.config import get_db_session_simple, close_session
from database.models import Patent, PatentChunk, DataSourceFile
from sqlalchemy import func, inspect

def examine_database_schema():
    """Examine the database schema to understand available columns."""
    print("🔍 Examining Database Schema")
    print("=" * 50)
    
    session = get_db_session_simple()
    
    try:
        # Get table metadata
        inspector = inspect(session.bind)
        
        # Patent table columns
        print("📊 Patent Table Columns:")
        patent_columns = inspector.get_columns('patents')
        for col in patent_columns:
            print(f"   📋 {col['name']}: {col['type']}")
        
        print("\n📊 Patent Chunk Table Columns:")
        chunk_columns = inspector.get_columns('patent_chunks')
        for col in chunk_columns:
            print(f"   📋 {col['name']}: {col['type']}")
        
        return patent_columns, chunk_columns
        
    except Exception as e:
        print(f"❌ Error examining schema: {e}")
        return [], []
    
    finally:
        close_session(session)

def analyze_sample_data():
    """Analyze sample data to understand field formats."""
    print("\n🔬 Analyzing Sample Data")
    print("=" * 30)
    
    session = get_db_session_simple()
    
    try:
        # Get a few sample patents
        sample_patents = session.query(Patent).limit(5).all()
        
        for i, patent in enumerate(sample_patents, 1):
            print(f"\n📄 Sample Patent {i}: {patent.publication_number}")
            print(f"   📝 Title: {patent.title_en[:60] if patent.title_en else 'N/A'}...")
            print(f"   🌍 Country: {patent.publication_country}")
            print(f"   📅 Date: {patent.publication_date}")
            
            # Check SDG fields
            if patent.sdg_number:
                try:
                    sdg_data = json.loads(patent.sdg_number) if isinstance(patent.sdg_number, str) else patent.sdg_number
                    print(f"   🎯 SDG Numbers: {sdg_data}")
                except:
                    print(f"   🎯 SDG Numbers (raw): {patent.sdg_number}")
            
            # Check technology fields
            if patent.sdg_technology_fields:
                try:
                    tech_data = json.loads(patent.sdg_technology_fields) if isinstance(patent.sdg_technology_fields, str) else patent.sdg_technology_fields
                    print(f"   🔧 Tech Fields: {tech_data}")
                except:
                    print(f"   🔧 Tech Fields (raw): {patent.sdg_technology_fields}")
            
            # Check applicant countries
            if patent.applicant_countries:
                try:
                    countries_data = json.loads(patent.applicant_countries) if isinstance(patent.applicant_countries, str) else patent.applicant_countries
                    print(f"   🏢 Applicant Countries: {countries_data}")
                except:
                    print(f"   🏢 Applicant Countries (raw): {patent.applicant_countries}")
            
            # Check IPC technologies
            if patent.ipc_technologies:
                try:
                    ipc_data = json.loads(patent.ipc_technologies) if isinstance(patent.ipc_technologies, str) else patent.ipc_technologies
                    print(f"   ⚙️  IPC Technologies: {ipc_data}")
                except:
                    print(f"   ⚙️  IPC Technologies (raw): {patent.ipc_technologies}")
        
    except Exception as e:
        print(f"❌ Error analyzing sample data: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        close_session(session)

def test_prompt_queries():
    """Test each specific prompt to see if it can be answered."""
    print("\n🧪 Testing Prompt Queries")
    print("=" * 40)
    
    session = get_db_session_simple()
    
    try:
        prompts_and_queries = [
            {
                "prompt": "How many patents do you have?",
                "description": "Total count of patents in database",
                "test_query": lambda s: s.query(Patent).count()
            },
            {
                "prompt": "How many patents are in SDG 6?",
                "description": "Patents containing SDG 6 in their sdg_number field",
                "test_query": lambda s: s.query(Patent).filter(Patent.sdg_number.like('%6%')).count()
            },
            {
                "prompt": "How many technologies are under SDG 7?",
                "description": "Unique technology fields for patents with SDG 7",
                "test_query": lambda s: len(set([
                    tech for patent in s.query(Patent).filter(Patent.sdg_number.like('%7%')).all()
                    if patent.sdg_technology_fields
                    for tech in (json.loads(patent.sdg_technology_fields) if isinstance(patent.sdg_technology_fields, str) else patent.sdg_technology_fields or [])
                    if isinstance(tech, str)
                ]))
            },
            {
                "prompt": "How many patents are submitted from USA?",
                "description": "Patents with USA in applicant_countries field",
                "test_query": lambda s: s.query(Patent).filter(Patent.applicant_countries.like('%USA%')).count()
            },
            {
                "prompt": "How many SDG technology fields are in SDG 6?",
                "description": "Unique SDG technology fields for SDG 6 patents",
                "test_query": lambda s: len(set([
                    tech for patent in s.query(Patent).filter(Patent.sdg_number.like('%6%')).all()
                    if patent.sdg_technology_fields
                    for tech in (json.loads(patent.sdg_technology_fields) if isinstance(patent.sdg_technology_fields, str) else patent.sdg_technology_fields or [])
                    if isinstance(tech, str)
                ]))
            },
            {
                "prompt": "How many patents are in solar technology?",
                "description": "Patents containing 'solar' in technology-related fields",
                "test_query": lambda s: s.query(Patent).filter(
                    (Patent.ipc_technologies.like('%solar%')) |
                    (Patent.sdg_technology_fields.like('%solar%')) |
                    (Patent.title_en.like('%solar%')) |
                    (Patent.abstract_text.like('%solar%'))
                ).count()
            }
        ]
        
        results = []
        
        for test in prompts_and_queries:
            print(f"\n❓ Testing: '{test['prompt']}'")
            print(f"   📋 Description: {test['description']}")
            
            try:
                result = test['test_query'](session)
                print(f"   ✅ Result: {result:,}")
                
                # Verify the result makes sense
                if result >= 0:
                    results.append({
                        'prompt': test['prompt'],
                        'result': result,
                        'status': 'success'
                    })
                else:
                    results.append({
                        'prompt': test['prompt'],
                        'result': result,
                        'status': 'error - negative result'
                    })
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'prompt': test['prompt'],
                    'result': None,
                    'status': f'error - {str(e)}'
                })
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing queries: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        close_session(session)

def analyze_data_quality():
    """Analyze data quality for the specific fields used in prompts."""
    print("\n📊 Data Quality Analysis")
    print("=" * 30)
    
    session = get_db_session_simple()
    
    try:
        total_patents = session.query(Patent).count()
        print(f"📈 Total Patents: {total_patents:,}")
        
        # Check field completeness
        fields_to_check = [
            ('sdg_number', 'SDG Numbers'),
            ('sdg_technology_fields', 'SDG Technology Fields'),
            ('applicant_countries', 'Applicant Countries'),
            ('ipc_technologies', 'IPC Technologies'),
            ('title_en', 'English Titles'),
            ('abstract_text', 'Abstract Text')
        ]
        
        for field_name, field_label in fields_to_check:
            non_null_count = session.query(Patent).filter(getattr(Patent, field_name).isnot(None)).count()
            non_empty_count = session.query(Patent).filter(
                getattr(Patent, field_name).isnot(None) & 
                (getattr(Patent, field_name) != '')
            ).count()
            
            percentage = (non_empty_count / total_patents) * 100 if total_patents > 0 else 0
            print(f"   📋 {field_label}: {non_empty_count:,}/{total_patents:,} ({percentage:.1f}%)")
        
        # Sample some specific SDG and country data
        print(f"\n🔍 Sample Data Analysis:")
        
        # SDG 6 patents
        sdg6_count = session.query(Patent).filter(Patent.sdg_number.like('%6%')).count()
        print(f"   🎯 Patents with SDG 6: {sdg6_count:,}")
        
        # SDG 7 patents
        sdg7_count = session.query(Patent).filter(Patent.sdg_number.like('%7%')).count()
        print(f"   🎯 Patents with SDG 7: {sdg7_count:,}")
        
        # USA patents
        usa_count = session.query(Patent).filter(Patent.applicant_countries.like('%USA%')).count()
        print(f"   🇺🇸 Patents from USA: {usa_count:,}")
        
        # Solar patents
        solar_count = session.query(Patent).filter(
            (Patent.ipc_technologies.like('%solar%')) |
            (Patent.sdg_technology_fields.like('%solar%')) |
            (Patent.title_en.like('%solar%')) |
            (Patent.abstract_text.like('%solar%'))
        ).count()
        print(f"   ☀️ Solar-related patents: {solar_count:,}")
        
    except Exception as e:
        print(f"❌ Error analyzing data quality: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        close_session(session)

def main():
    """Main function to run all verification tests."""
    print("🔬 Patent Database Query Verification")
    print("=" * 50)
    
    # Examine schema
    examine_database_schema()
    
    # Analyze sample data
    analyze_sample_data()
    
    # Test prompt queries
    results = test_prompt_queries()
    
    # Analyze data quality
    analyze_data_quality()
    
    # Summary
    print("\n📋 Summary of Prompt Verification")
    print("=" * 40)
    
    if results:
        success_count = len([r for r in results if r['status'] == 'success'])
        total_count = len(results)
        
        print(f"✅ Successfully answerable prompts: {success_count}/{total_count}")
        
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status_icon} '{result['prompt']}' - {result['status']}")
    
    print("\n🎉 Verification completed!")

if __name__ == "__main__":
    main()
