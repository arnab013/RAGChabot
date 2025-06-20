#!/usr/bin/env python3
"""
Generate sample patent dataset for public repository
Creates 100 rows of realistic but fictional patent data
"""

import pandas as pd
import json
import random
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker for generating realistic but fake data
fake = Faker()

# Sample data templates
SAMPLE_TECHNOLOGIES = [
    "solar energy conversion", "battery technology", "wind power generation", 
    "water purification", "waste recycling", "electric vehicles", "energy storage",
    "renewable energy", "smart grid technology", "carbon capture", "fuel cells",
    "photovoltaic cells", "energy efficiency", "sustainable materials", "green chemistry"
]

SAMPLE_IPC_CODES = [
    "H01L31/00", "H02J3/00", "C02F1/00", "B29B17/00", "B60L11/00",
    "H01M10/00", "F03D9/00", "C08J11/00", "H02S10/00", "B01D53/00"
]

SAMPLE_COUNTRIES = ["US", "EP", "JP", "CN", "KR", "DE", "GB", "FR", "CA", "AU"]

SAMPLE_APPLICANTS = [
    "GreenTech Solutions Inc.", "Sustainable Energy Corp.", "EcoInnovation Ltd.",
    "Future Power Systems", "Clean Technology Partners", "Renewable Resources Co.",
    "Advanced Materials Inc.", "Environmental Solutions LLC", "Smart Energy Group",
    "NextGen Technologies", "Global Sustainability Corp.", "Innovation Labs Ltd."
]

SAMPLE_INVENTORS = [
    "John Smith", "Maria Garcia", "David Chen", "Sarah Johnson", "Michael Brown",
    "Lisa Wang", "Robert Davis", "Anna Mueller", "James Wilson", "Elena Rodriguez",
    "Thomas Anderson", "Jennifer Lee", "Mark Thompson", "Sophie Martin", "Alex Kim"
]

SDG_NUMBERS = [7, 9, 11, 12, 13, 14, 15]  # Common SDGs for patent data

def generate_sample_patent_data(num_records=100):
    """Generate sample patent data"""
    patents = []
    
    for i in range(num_records):
        # Generate publication number
        country = random.choice(SAMPLE_COUNTRIES)
        number = f"{random.randint(1000000, 9999999)}"
        kind = random.choice(["A1", "B1", "B2"])
        pub_number = f"{country}{number}{kind}"
        
        # Generate dates
        pub_date = fake.date_between(start_date=datetime(2015, 1, 1), end_date=datetime(2024, 12, 31))
        
        # Technology selection
        tech = random.choice(SAMPLE_TECHNOLOGIES)
        
        # Generate patent data
        patent = {
            'publication_number': pub_number,
            'publication_country': country,
            'publication_kind': kind,
            'publication_date': pub_date.strftime('%Y-%m-%d'),
            'ipc': json.dumps([random.choice(SAMPLE_IPC_CODES) for _ in range(random.randint(1, 3))]),
            'title_en': f"Method and System for {tech.title()} - Patent {i+1}",
            'abstract_text': f"This invention relates to {tech} and provides an improved method for implementing sustainable solutions. The system comprises novel components that enhance efficiency and reduce environmental impact. The invention addresses key challenges in the field and offers significant advantages over prior art solutions.",
            'sdg_number': json.dumps([random.choice(SDG_NUMBERS)]),
            'analysis_explanation': json.dumps({
                "summary": f"This patent contributes to sustainable development through {tech}",
                "impact": "Positive environmental and economic benefits",
                "application": "Industrial and commercial applications"
            }),
            'applicant_names': json.dumps([random.choice(SAMPLE_APPLICANTS)]),
            'applicant_countries': json.dumps([country]),
            'applicant_count': 1,
            'inventor_names': json.dumps([
                random.choice(SAMPLE_INVENTORS) for _ in range(random.randint(1, 4))
            ]),
            'inventor_countries': json.dumps([country]),
            'inventor_count': random.randint(1, 4),
            'ipc_tech_field': json.dumps([f"Technology Field {random.randint(1, 10)}"]),
            'ipc_technologies': json.dumps([tech]),
            'sdg_technology_fields': json.dumps([f"SDG Technology {random.randint(1, 5)}"]),
            'analysis_potential_beneficiaries': json.dumps([
                "Industry", "Consumers", "Environment", "Society"
            ]),
            'designated_states_contracting': json.dumps([
                random.choice(SAMPLE_COUNTRIES) for _ in range(random.randint(2, 8))
            ]),
            'designated_states_extension': json.dumps([]),
            'designated_states_validation': json.dumps([]),
            'prior_art': json.dumps([]),
            'reference': json.dumps([]),
            'parent': json.dumps([]),
            'pct_publication_number': f"PCT/{country}{random.randint(2020, 2024)}/{random.randint(100000, 999999)}",
            'parent_publication_number': ""
        }
        
        patents.append(patent)
    
    return patents

def main():
    """Generate and save sample patent data"""
    print("Generating sample patent dataset...")
    
    # Generate sample data
    patent_data = generate_sample_patent_data(100)
    
    # Create DataFrame
    df = pd.DataFrame(patent_data)
    
    # Save to Excel
    output_file = "data/sample_patent_data.xlsx"
    df.to_excel(output_file, index=False, sheet_name="Patents")
    
    print(f"Sample dataset created: {output_file}")
    print(f"Generated {len(patent_data)} patent records")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Display first few rows
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()
