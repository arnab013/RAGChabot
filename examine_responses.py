#!/usr/bin/env python3
"""
Examine responses in detail to identify issues with expected vs actual data
"""

import requests
import json
import time

# Test queries focusing on analytics and chart queries (skipping Patent Search)
TEST_QUERIES = {
    "Bar Charts & Analytics": [
        "Show me patent publication trends by year",
        "Which SDG has the most patents?", 
        "Show top 10 inventors by patent count"
    ],
    "Line Charts & Trends": [
        "Show patent publication trends in last 12 months",
        "Display publication trends for the last 6 months"
    ],
    "Pie Charts & Distribution": [
        "Show percentage distribution of patents by SDG",
        "What's the proportion of patents by technology type?"
    ]
}

API_BASE_URL = "http://localhost:5000"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/search"

def detailed_query_analysis(category, query):
    """Analyze a query response in detail"""
    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING: [{category}] {query}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        response = requests.post(CHAT_ENDPOINT, 
                               json={"query": query}, 
                               timeout=60,
                               headers={'Content-Type': 'application/json'})
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
        print(f"📊 HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Print full response structure
            print(f"\n📋 RESPONSE STRUCTURE:")
            print(f"Keys: {list(data.keys())}")
            
            # Analyze message
            message = data.get('message', '')
            print(f"\n💬 MESSAGE:")
            print(f"Length: {len(message)} characters")
            print(f"Content: {message}")
            
            # Analyze chart data
            chart = data.get('chart')
            print(f"\n📈 CHART DATA:")
            if chart:
                print(f"Chart Type: {chart.get('type', 'Not specified')}")
                print(f"Chart Keys: {list(chart.keys())}")
                
                # Check chart data structure
                if 'data' in chart:
                    chart_data = chart['data']
                    print(f"Data Type: {type(chart_data)}")
                    if isinstance(chart_data, list) and chart_data:
                        print(f"Data Length: {len(chart_data)}")
                        print(f"First Item: {chart_data[0]}")
                        print(f"Sample Data: {chart_data[:3]}")
                    elif isinstance(chart_data, dict):
                        print(f"Data Structure: {list(chart_data.keys())}")
                        print(f"Data Content: {chart_data}")
                else:
                    print("No 'data' field in chart")
                
                # Check labels
                if 'labels' in chart:
                    labels = chart['labels']
                    print(f"Labels: {labels[:5] if isinstance(labels, list) else labels}")
                
                # Check other chart properties
                for key in ['title', 'xLabel', 'yLabel', 'backgroundColor', 'borderColor']:
                    if key in chart:
                        print(f"{key}: {chart[key]}")
            else:
                print("No chart data provided")
            
            # Check for other fields
            other_fields = ['error', 'insight', 'takeaway']
            for field in other_fields:
                if field in data and data[field]:
                    print(f"\n{field.upper()}: {data[field]}")
                    
            return True
            
        else:
            print(f"❌ FAILED: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("🔍 DETAILED RESPONSE ANALYSIS")
    print("Examining what data is returned vs what's expected")
    print("Skipping Patent Search section as requested")
    
    for category, queries in TEST_QUERIES.items():
        print(f"\n\n🏷️  CATEGORY: {category}")
        
        for query in queries:
            detailed_query_analysis(category, query)
            time.sleep(2)  # Small delay between requests
            
            # Ask user if they want to continue after each query
            print(f"\n{'='*40}")
            
        print(f"\n✅ Completed category: {category}")

if __name__ == "__main__":
    main()
