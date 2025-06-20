#!/usr/bin/env python3
"""
Sample data loader and basic analysis
Demonstrates how to work with the sample patent dataset
"""

import pandas as pd
import json
import sys
import os

def load_sample_data():
    """Load the sample patent dataset"""
    data_file = "data/sample_patent_data.xlsx"
    
    if not os.path.exists(data_file):
        print(f"Sample data file not found: {data_file}")
        print("Run 'python generate_sample_data.py' to create sample data")
        return None
    
    try:
        df = pd.read_excel(data_file)
        print(f"Successfully loaded {len(df)} patent records")
        return df
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None

def analyze_sample_data(df):
    """Perform basic analysis of the sample dataset"""
    print("\n" + "="*50)
    print("SAMPLE PATENT DATASET ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"\nTotal Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Publication countries
    countries = df['publication_country'].value_counts()
    print(f"\nPublication Countries:")
    for country, count in countries.items():
        print(f"  {country}: {count} patents")
    
    # Publication years
    df['pub_year'] = pd.to_datetime(df['publication_date']).dt.year
    years = df['pub_year'].value_counts().sort_index()
    print(f"\nPublication Years:")
    for year, count in years.items():
        print(f"  {year}: {count} patents")
    
    # SDG distribution
    sdg_counts = {}
    for sdg_str in df['sdg_number']:
        sdgs = json.loads(sdg_str)
        for sdg in sdgs:
            sdg_counts[sdg] = sdg_counts.get(sdg, 0) + 1
    
    print(f"\nSDG Distribution:")
    for sdg, count in sorted(sdg_counts.items()):
        print(f"  SDG {sdg}: {count} patents")
    
    # Technology fields
    tech_counts = {}
    for tech_str in df['ipc_technologies']:
        techs = json.loads(tech_str)
        for tech in techs:
            tech_counts[tech] = tech_counts.get(tech, 0) + 1
    
    print(f"\nTop Technologies:")
    sorted_techs = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)
    for tech, count in sorted_techs[:10]:
        print(f"  {tech}: {count} patents")
    
    # Sample patent details
    print(f"\n" + "="*50)
    print("SAMPLE PATENT DETAILS")
    print("="*50)
    
    sample_patent = df.iloc[0]
    print(f"\nPublication Number: {sample_patent['publication_number']}")
    print(f"Title: {sample_patent['title_en']}")
    print(f"Country: {sample_patent['publication_country']}")
    print(f"Date: {sample_patent['publication_date']}")
    print(f"Applicant: {json.loads(sample_patent['applicant_names'])[0]}")
    print(f"Inventors: {', '.join(json.loads(sample_patent['inventor_names']))}")
    print(f"Technologies: {', '.join(json.loads(sample_patent['ipc_technologies']))}")
    print(f"SDG: {json.loads(sample_patent['sdg_number'])}")
    
    print(f"\nAbstract:")
    print(f"{sample_patent['abstract_text'][:200]}...")

def main():
    """Main function"""
    print("Patent Research Platform - Sample Data Analysis")
    
    # Load sample data
    df = load_sample_data()
    if df is None:
        sys.exit(1)
    
    # Analyze the data
    analyze_sample_data(df)
    
    print(f"\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("\n1. Import this data into the database system")
    print("2. Process and create embeddings for semantic search")
    print("3. Test the web interface with sample queries")
    print("4. Replace with real patent data for production use")
    print("\nFor more information, see data/README.md")

if __name__ == "__main__":
    main()
