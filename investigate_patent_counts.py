#!/usr/bin/env python3
"""
Investigate patent count discrepancies in the database
"""

import sqlite3
import json
from collections import Counter

# Connect to the database
db_path = "data/patents.db"

def check_patent_counts():
    """Check various patent counts to identify discrepancies"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 INVESTIGATING PATENT COUNT DISCREPANCIES")
    print("=" * 60)
    
    # 1. Total patents in database
    cursor.execute("SELECT COUNT(*) FROM patents")
    total_patents = cursor.fetchone()[0]
    print(f"📊 Total patents in database: {total_patents:,}")
      # 2. Patents with SDG classifications
    cursor.execute("SELECT COUNT(*) FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != ''")
    sdg_patents = cursor.fetchone()[0]
    print(f"🎯 Patents with SDG data: {sdg_patents:,}")
    
    # 3. Patents with IPC classifications  
    cursor.execute("SELECT COUNT(*) FROM patents WHERE ipc IS NOT NULL AND ipc != ''")
    ipc_patents = cursor.fetchone()[0]
    print(f"🏷️  Patents with IPC data: {ipc_patents:,}")
    
    # 4. Let's check what the SDG count actually shows
    cursor.execute("SELECT sdg_number FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != ''")
    sdg_data = cursor.fetchall()
    
    sdg_counter = Counter()
    valid_sdg_count = 0
    
    for row in sdg_data:
        sdg_value = row[0]
        if sdg_value:
            try:
                if isinstance(sdg_value, str):
                    sdg_list = json.loads(sdg_value)
                else:
                    sdg_list = sdg_value
                    
                if isinstance(sdg_list, list) and sdg_list:
                    valid_sdg_count += 1
                    for sdg in sdg_list:
                        if isinstance(sdg, (int, str)):
                            sdg_counter[f"SDG {sdg}"] += 1
            except (json.JSONDecodeError, TypeError):
                continue
    
    print(f"✅ Patents with valid SDG classifications: {valid_sdg_count:,}")
    print(f"📈 Total SDG entries counted: {sum(sdg_counter.values()):,}")
    
    # 5. Check IPC count details
    cursor.execute("SELECT ipc FROM patents WHERE ipc IS NOT NULL AND ipc != ''")
    ipc_data = cursor.fetchall()
    
    ipc_counter = Counter()
    valid_ipc_count = 0
    
    for row in ipc_data:
        ipc_value = row[0]
        if ipc_value:
            try:
                if isinstance(ipc_value, str):
                    ipc_list = json.loads(ipc_value)
                else:
                    ipc_list = ipc_value
                    
                if isinstance(ipc_list, list) and ipc_list:
                    valid_ipc_count += 1
                    for ipc in ipc_list:
                        if isinstance(ipc, str) and len(ipc) >= 1:
                            main_section = ipc[0].upper()
                            ipc_counter[main_section] += 1
            except (json.JSONDecodeError, TypeError):
                continue
    
    print(f"✅ Patents with valid IPC classifications: {valid_ipc_count:,}")
    print(f"📈 Total IPC entries counted: {sum(ipc_counter.values()):,}")
      # 6. Show sample data to understand structure
    print(f"\n📋 SAMPLE DATA ANALYSIS")
    print("=" * 30)
    
    cursor.execute("SELECT publication_number, sdg_number, ipc FROM patents LIMIT 5")
    samples = cursor.fetchall()
    
    for i, (pub_num, sdg, ipc) in enumerate(samples, 1):
        print(f"\nSample {i}: {pub_num}")
        print(f"  SDG: {sdg[:100] if sdg else 'None'}...")
        print(f"  IPC: {ipc[:100] if ipc else 'None'}...")
      # 7. Check for null/empty patterns
    print(f"\n🔍 NULL/EMPTY PATTERNS")
    print("=" * 25)
    
    cursor.execute("SELECT COUNT(*) FROM patents WHERE sdg_number IS NULL")
    sdg_null = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patents WHERE sdg_number = ''")
    sdg_empty = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patents WHERE ipc IS NULL")
    ipc_null = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patents WHERE ipc = ''")
    ipc_empty = cursor.fetchone()[0]
    
    print(f"SDG NULL: {sdg_null:,}, SDG Empty: {sdg_empty:,}")
    print(f"IPC NULL: {ipc_null:,}, IPC Empty: {ipc_empty:,}")
    
    # 8. Show top SDGs and IPCs
    print(f"\n📊 TOP SDGs")
    print("=" * 15)
    for sdg, count in sdg_counter.most_common(10):
        print(f"  {sdg}: {count:,}")
    
    print(f"\n📊 TOP IPC SECTIONS")  
    print("=" * 20)
    for ipc, count in ipc_counter.most_common(10):
        print(f"  {ipc}: {count:,}")
    
    # 9. Summary of discrepancies
    print(f"\n🚨 DISCREPANCY SUMMARY")
    print("=" * 25)
    print(f"Total patents in DB: {total_patents:,}")
    print(f"Patents with SDG data: {sdg_patents:,} ({(sdg_patents/total_patents*100):.1f}%)")
    print(f"Valid SDG classified: {valid_sdg_count:,} ({(valid_sdg_count/total_patents*100):.1f}%)")
    print(f"Patents with IPC data: {ipc_patents:,} ({(ipc_patents/total_patents*100):.1f}%)")
    print(f"Valid IPC classified: {valid_ipc_count:,} ({(valid_ipc_count/total_patents*100):.1f}%)")
    
    conn.close()

if __name__ == "__main__":
    check_patent_counts()
