#!/usr/bin/env python3
"""
Verify the correct patent counting logic
"""

import sqlite3
import json
from collections import Counter

# Connect to the database
db_path = "data/patents.db"

def verify_patent_counting():
    """Verify the correct way to count patents vs classifications"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 VERIFYING PATENT COUNTING LOGIC")
    print("=" * 50)
    
    # 1. Total unique patents in database
    cursor.execute("SELECT COUNT(DISTINCT publication_number) FROM patents")
    total_unique_patents = cursor.fetchone()[0]
    print(f"📊 Total unique patents: {total_unique_patents:,}")
    
    # 2. Unique patents with SDG data
    cursor.execute("SELECT COUNT(DISTINCT publication_number) FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != ''")
    unique_patents_with_sdg = cursor.fetchone()[0]
    print(f"🎯 Unique patents with SDG data: {unique_patents_with_sdg:,}")
    
    # 3. Unique patents with IPC data
    cursor.execute("SELECT COUNT(DISTINCT publication_number) FROM patents WHERE ipc IS NOT NULL AND ipc != ''")
    unique_patents_with_ipc = cursor.fetchone()[0]
    print(f"🏷️  Unique patents with IPC data: {unique_patents_with_ipc:,}")
    
    # 4. Sample patent data to show multiple classifications
    print(f"\n📋 SAMPLE PATENTS WITH MULTIPLE CLASSIFICATIONS:")
    print("=" * 55)
    
    cursor.execute("""
        SELECT publication_number, sdg_number, ipc 
        FROM patents 
        WHERE sdg_number IS NOT NULL AND ipc IS NOT NULL 
        LIMIT 5
    """)
    samples = cursor.fetchall()
    
    for i, (pub_num, sdg, ipc) in enumerate(samples, 1):
        print(f"\nPatent {i}: {pub_num}")
        
        # Parse SDG data
        try:
            if sdg:
                sdg_list = json.loads(sdg) if isinstance(sdg, str) else sdg
                sdg_count = len(sdg_list) if isinstance(sdg_list, list) else 0
                print(f"  SDG Numbers: {sdg_list} ({sdg_count} SDGs)")
            else:
                print(f"  SDG Numbers: None")
        except:
            print(f"  SDG Numbers: Error parsing")
        
        # Parse IPC data
        try:
            if ipc:
                ipc_list = json.loads(ipc) if isinstance(ipc, str) else ipc
                ipc_count = len(ipc_list) if isinstance(ipc_list, list) else 0
                print(f"  IPC Codes: {ipc_list} ({ipc_count} IPCs)")
            else:
                print(f"  IPC Codes: None")
        except:
            print(f"  IPC Codes: Error parsing")
    
    # 5. Count total classifications vs unique patents
    print(f"\n📊 CLASSIFICATION COUNTS VS PATENT COUNTS:")
    print("=" * 50)
    
    # SDG classification count
    cursor.execute("SELECT sdg_number FROM patents WHERE sdg_number IS NOT NULL AND sdg_number != ''")
    sdg_data = cursor.fetchall()
    
    total_sdg_classifications = 0
    for row in sdg_data:
        try:
            sdg_list = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            if isinstance(sdg_list, list):
                total_sdg_classifications += len(sdg_list)
        except:
            continue
    
    # IPC classification count
    cursor.execute("SELECT ipc FROM patents WHERE ipc IS NOT NULL AND ipc != ''")
    ipc_data = cursor.fetchall()
    
    total_ipc_classifications = 0
    for row in ipc_data:
        try:
            ipc_list = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            if isinstance(ipc_list, list):
                total_ipc_classifications += len(ipc_list)
        except:
            continue
    
    print(f"SDG Classifications:")
    print(f"  • Total SDG entries: {total_sdg_classifications:,}")
    print(f"  • Unique patents with SDGs: {unique_patents_with_sdg:,}")
    print(f"  • Average SDGs per patent: {total_sdg_classifications/unique_patents_with_sdg:.1f}")
    
    print(f"\nIPC Classifications:")
    print(f"  • Total IPC entries: {total_ipc_classifications:,}")
    print(f"  • Unique patents with IPCs: {unique_patents_with_ipc:,}")
    print(f"  • Average IPCs per patent: {total_ipc_classifications/unique_patents_with_ipc:.1f}")
    
    # 6. Recommendation for correct labeling
    print(f"\n💡 RECOMMENDED CORRECT LABELING:")
    print("=" * 35)
    print(f"❌ WRONG: 'Total SDG-Classified Patents: {total_sdg_classifications:,}'")
    print(f"✅ RIGHT: 'Patents with SDG Classifications: {unique_patents_with_sdg:,}'")
    print(f"✅ RIGHT: 'Total SDG Classifications: {total_sdg_classifications:,}'")
    print()
    print(f"❌ WRONG: 'Total Classified Patents: {total_ipc_classifications:,}'")
    print(f"✅ RIGHT: 'Patents with IPC Classifications: {unique_patents_with_ipc:,}'")
    print(f"✅ RIGHT: 'Total IPC Classifications: {total_ipc_classifications:,}'")
    
    conn.close()

if __name__ == "__main__":
    verify_patent_counting()
