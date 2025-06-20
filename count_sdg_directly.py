#!/usr/bin/env python3
"""
Count SDG numbers directly from the database to understand the real distribution
"""

import sqlite3
import json
from collections import Counter

def count_sdg_numbers_directly():
    """Count SDG numbers directly from the database"""
    
    # Connect to the database
    conn = sqlite3.connect("data/patents.db")
    cursor = conn.cursor()
    
    print("🔍 DIRECT SDG COUNT FROM DATABASE")
    print("=" * 50)
    
    # Get all patents with their SDG numbers
    cursor.execute("SELECT publication_number, sdg_number FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != ''")
    patents_with_sdg = cursor.fetchall()
    
    print(f"📊 Patents with SDG data: {len(patents_with_sdg):,}")
    
    # Count unique patents vs total SDG entries
    unique_patents = set()
    sdg_counter = Counter()
    patents_by_sdg = {}  # SDG -> list of patent numbers
    
    valid_sdg_patents = 0
    total_sdg_entries = 0
    
    for pub_num, sdg_data in patents_with_sdg:
        if sdg_data:
            try:
                # Parse SDG data (it might be JSON string or already parsed)
                if isinstance(sdg_data, str):
                    if sdg_data.isdigit():
                        # Single SDG number as string
                        sdg_list = [int(sdg_data)]
                    else:
                        # Try to parse as JSON
                        try:
                            sdg_list = json.loads(sdg_data)
                            if not isinstance(sdg_list, list):
                                sdg_list = [sdg_list]
                        except json.JSONDecodeError:
                            # Might be comma-separated values
                            sdg_list = [int(x.strip()) for x in sdg_data.split(',') if x.strip().isdigit()]
                else:
                    sdg_list = [sdg_data] if not isinstance(sdg_data, list) else sdg_data
                
                # Process valid SDG numbers
                if sdg_list:
                    unique_patents.add(pub_num)
                    valid_sdg_patents += 1
                    
                    for sdg in sdg_list:
                        if isinstance(sdg, (int, str)) and str(sdg).isdigit():
                            sdg_num = int(sdg)
                            if 1 <= sdg_num <= 17:  # Valid SDG range
                                sdg_counter[sdg_num] += 1
                                total_sdg_entries += 1
                                
                                # Track which patents belong to each SDG
                                if sdg_num not in patents_by_sdg:
                                    patents_by_sdg[sdg_num] = []
                                patents_by_sdg[sdg_num].append(pub_num)
            
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                print(f"Error parsing SDG data for patent {pub_num}: {sdg_data} -> {e}")
                continue
    
    print(f"✅ Unique patents with valid SDGs: {len(unique_patents):,}")
    print(f"📈 Total SDG classifications: {total_sdg_entries:,}")
    print(f"📊 Average SDGs per patent: {total_sdg_entries/len(unique_patents):.2f}")
    
    print(f"\n📋 SDG DISTRIBUTION:")
    print("=" * 30)
    print(f"{'SDG':<5} {'Classifications':<15} {'Unique Patents':<15} {'%':<8}")
    print("-" * 45)
    
    for sdg in sorted(sdg_counter.keys()):
        classification_count = sdg_counter[sdg]
        unique_patent_count = len(set(patents_by_sdg[sdg]))
        percentage = (classification_count / total_sdg_entries * 100) if total_sdg_entries > 0 else 0
        
        print(f"{sdg:<5} {classification_count:<15} {unique_patent_count:<15} {percentage:<7.1f}%")
    
    print(f"\n🔍 SAMPLE ANALYSIS:")
    print("=" * 20)
    
    # Show some examples of patents with multiple SDGs
    multi_sdg_examples = []
    for pub_num, sdg_data in patents_with_sdg[:20]:  # Check first 20
        if sdg_data:
            try:
                if isinstance(sdg_data, str):
                    if ',' in sdg_data or '[' in sdg_data:
                        # This might be multiple SDGs
                        if sdg_data.isdigit():
                            sdg_list = [int(sdg_data)]
                        else:
                            try:
                                sdg_list = json.loads(sdg_data)
                                if not isinstance(sdg_list, list):
                                    sdg_list = [sdg_list]
                            except:
                                sdg_list = [int(x.strip()) for x in sdg_data.split(',') if x.strip().isdigit()]
                    else:
                        sdg_list = [int(sdg_data)] if sdg_data.isdigit() else []
                else:
                    sdg_list = [sdg_data] if not isinstance(sdg_data, list) else sdg_data
                
                if len(sdg_list) > 1:
                    multi_sdg_examples.append((pub_num, sdg_list))
                    if len(multi_sdg_examples) >= 5:
                        break
                        
            except:
                continue
    
    if multi_sdg_examples:
        print("Patents with multiple SDGs:")
        for pub_num, sdgs in multi_sdg_examples:
            print(f"  Patent {pub_num}: SDGs {sdgs}")
    else:
        print("No examples of patents with multiple SDGs found in sample")
    
    # Show raw data examples
    print(f"\n📄 RAW DATA SAMPLES:")
    print("=" * 25)
    cursor.execute("SELECT publication_number, sdg_number FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != '' LIMIT 10")
    samples = cursor.fetchall()
    
    for pub_num, sdg_raw in samples:
        print(f"Patent {pub_num}: '{sdg_raw}' (type: {type(sdg_raw).__name__})")
    
    # Summary
    print(f"\n🚨 SUMMARY:")
    print("=" * 15)
    print(f"• Database contains {len(patents_with_sdg):,} patents with SDG data")
    print(f"• {len(unique_patents):,} unique patents have valid SDG classifications")
    print(f"• {total_sdg_entries:,} total SDG classification entries")
    print(f"• This means the current system is counting CLASSIFICATIONS, not PATENTS")
    print(f"• Each patent averages {total_sdg_entries/len(unique_patents):.2f} SDG classifications")
    
    conn.close()

if __name__ == "__main__":
    count_sdg_numbers_directly()
