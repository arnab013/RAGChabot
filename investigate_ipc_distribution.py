#!/usr/bin/env python3
"""
Investigate IPC distribution issues directly from the database
"""

import sqlite3
import json
from collections import Counter

def investigate_ipc_distribution():
    """Investigate IPC distribution issues directly from database"""
    
    # Connect to the database
    conn = sqlite3.connect("data/patents.db")
    cursor = conn.cursor()
    
    print("🔍 INVESTIGATING IPC DISTRIBUTION ISSUES")
    print("=" * 50)
    
    # Get all patents with their IPC data
    cursor.execute("SELECT publication_number, ipc FROM patents WHERE ipc IS NOT NULL AND ipc != ''")
    patents_with_ipc = cursor.fetchall()
    
    print(f"📊 Patents with IPC data: {len(patents_with_ipc):,}")
    
    # Count unique patents vs total IPC entries
    unique_patents = set()
    ipc_counter = Counter()
    patents_by_ipc = {}  # IPC section -> list of patent numbers
    
    valid_ipc_patents = 0
    total_ipc_entries = 0
    parse_errors = 0
    
    print(f"\n📋 SAMPLE IPC DATA:")
    print("=" * 25)
    
    # Show some raw IPC data examples
    for i, (pub_num, ipc_data) in enumerate(patents_with_ipc[:10]):
        print(f"Patent {pub_num}: '{ipc_data[:100]}...' (type: {type(ipc_data).__name__})")
    
    print(f"\n🔍 PROCESSING IPC DATA:")
    print("=" * 30)
    
    for pub_num, ipc_data in patents_with_ipc:
        if ipc_data:
            try:
                # Parse IPC data (should be JSON array)
                if isinstance(ipc_data, str):
                    ipc_list = json.loads(ipc_data)
                else:
                    ipc_list = ipc_data
                
                if isinstance(ipc_list, list) and ipc_list:
                    unique_patents.add(pub_num)
                    valid_ipc_patents += 1
                    
                    for ipc in ipc_list:
                        if isinstance(ipc, str) and len(ipc) >= 1:
                            # Extract main section (first letter)
                            main_section = ipc[0].upper()
                            if main_section.isalpha():  # Valid IPC section
                                ipc_counter[main_section] += 1
                                total_ipc_entries += 1
                                
                                # Track which patents belong to each IPC
                                if main_section not in patents_by_ipc:
                                    patents_by_ipc[main_section] = []
                                patents_by_ipc[main_section].append(pub_num)
                            else:
                                # Handle weird IPC codes
                                ipc_counter[main_section] += 1
                                total_ipc_entries += 1
                
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                parse_errors += 1
                if parse_errors <= 5:  # Show first 5 errors
                    print(f"Parse error for patent {pub_num}: {ipc_data[:50]}... -> {e}")
                continue
    
    print(f"✅ Unique patents with valid IPC: {len(unique_patents):,}")
    print(f"📈 Total IPC classifications: {total_ipc_entries:,}")
    print(f"📊 Average IPC codes per patent: {total_ipc_entries/len(unique_patents):.2f}")
    print(f"❌ Parse errors: {parse_errors:,}")
    
    print(f"\n📋 IPC SECTION DISTRIBUTION:")
    print("=" * 35)
    print(f"{'Section':<10} {'Classifications':<15} {'Unique Patents':<15} {'%':<8}")
    print("-" * 50)
    
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
    
    for section in sorted(ipc_counter.keys()):
        classification_count = ipc_counter[section]
        unique_patent_count = len(set(patents_by_ipc[section])) if section in patents_by_ipc else 0
        percentage = (classification_count / total_ipc_entries * 100) if total_ipc_entries > 0 else 0
        description = ipc_sections.get(section, 'Unknown')
        
        print(f"{section:<10} {classification_count:<15} {unique_patent_count:<15} {percentage:<7.1f}%")
        if section not in ipc_sections:
            print(f"  ⚠️  Unknown section: {section}")
    
    print(f"\n🔍 SAMPLE MULTI-IPC PATENTS:")
    print("=" * 30)
    
    # Show examples of patents with multiple IPC codes
    multi_ipc_examples = []
    for pub_num, ipc_data in patents_with_ipc[:50]:  # Check first 50
        if ipc_data:
            try:
                if isinstance(ipc_data, str):
                    ipc_list = json.loads(ipc_data)
                else:
                    ipc_list = ipc_data
                    
                if isinstance(ipc_list, list) and len(ipc_list) > 1:
                    multi_ipc_examples.append((pub_num, ipc_list))
                    if len(multi_ipc_examples) >= 5:
                        break
            except:
                continue
    
    if multi_ipc_examples:
        for pub_num, ipc_codes in multi_ipc_examples:
            sections = [code[0] if code else '?' for code in ipc_codes]
            print(f"Patent {pub_num}: {len(ipc_codes)} codes -> sections {sections}")
    
    print(f"\n🚨 SUMMARY:")
    print("=" * 15)
    print(f"• Database contains {len(patents_with_ipc):,} patents with IPC data")
    print(f"• {len(unique_patents):,} unique patents have valid IPC classifications") 
    print(f"• {total_ipc_entries:,} total IPC classification entries")
    print(f"• Average {total_ipc_entries/len(unique_patents):.2f} IPC codes per patent")
    print(f"• {parse_errors:,} patents had IPC parsing errors")
    
    # Now test what the current system returns
    print(f"\n🧪 TESTING CURRENT SYSTEM:")
    print("=" * 30)
    
    import requests
    try:
        response = requests.post("http://localhost:5000/api/search", 
                               json={"query": "What's the proportion of patents by technology type?"}, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            message = data.get('message', '')
            print("Current system response:")
            print(message[:500] + "..." if len(message) > 500 else message)
            
            # Extract total from response
            import re
            match = re.search(r"Total.*?(\d{1,3}(?:,\d{3})*)", message)
            if match:
                system_total = match.group(1).replace(',', '')
                print(f"\nSystem reports: {system_total}")
                print(f"Expected: {total_ipc_entries:,}")
                print(f"Match: {'✅ YES' if system_total == str(total_ipc_entries) else '❌ NO'}")
        else:
            print(f"API Error: {response.status_code}")
            
    except Exception as e:
        print(f"API Test Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    investigate_ipc_distribution()
