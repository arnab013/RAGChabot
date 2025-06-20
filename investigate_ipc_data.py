#!/usr/bin/env python3
"""
Direct database investigation of IPC data
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import sqlite3
import json
from collections import Counter

def check_ipc_database():
    """Check IPC data directly in the database"""
    print("=== Direct Database IPC Investigation ===\n")
    
    # Connect to database
    db_path = "data/patents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
      # 1. Check total patents with IPC data
    cursor.execute("SELECT COUNT(*) FROM patents WHERE ipc_technologies IS NOT NULL AND ipc_technologies != ''")
    total_patents_with_ipc = cursor.fetchone()[0]
    print(f"Total patents with IPC technologies data: {total_patents_with_ipc:,}")
    
    # 2. Check total patents
    cursor.execute("SELECT COUNT(*) FROM patents")
    total_patents = cursor.fetchone()[0]
    print(f"Total patents in database: {total_patents:,}")    # 3. Sample some IPC data to see the format
    print("\n=== Sample IPC Technologies Data ===")
    cursor.execute("SELECT publication_number, ipc_technologies FROM patents WHERE ipc_technologies IS NOT NULL AND ipc_technologies != '' LIMIT 10")
    samples = cursor.fetchall()
    
    for i, (patent_num, ipc_data) in enumerate(samples):
        print(f"\nSample {i+1}: Patent {patent_num}")
        print(f"IPC Technologies raw: {ipc_data[:200]}...")
        
        # Try to parse the IPC data
        try:
            if isinstance(ipc_data, str):
                parsed = json.loads(ipc_data)
                print(f"IPC Technologies parsed: {parsed}")
                if isinstance(parsed, list):
                    print(f"Number of IPC codes: {len(parsed)}")
                    for code in parsed[:3]:  # Show first 3
                        print(f"  - {code}")
        except:
            print(f"Could not parse as JSON: {ipc_data}")
      # 4. Count total IPC classifications using the same logic as the handler
    print(f"\n=== IPC Classification Count Analysis ===")
    cursor.execute("SELECT ipc_technologies FROM patents WHERE ipc_technologies IS NOT NULL AND ipc_technologies != ''")
    all_ipc_data = cursor.fetchall()
    
    ipc_counter = Counter()
    total_classifications = 0
    patents_processed = 0
    error_count = 0
    
    for (ipc_data,) in all_ipc_data:
        patents_processed += 1
        if ipc_data:
            try:
                # Parse IPC classifications (same logic as handler)
                parsed_ipc = json.loads(ipc_data) if isinstance(ipc_data, str) else ipc_data
                if isinstance(parsed_ipc, list):
                    for ipc in parsed_ipc:
                        if isinstance(ipc, str) and len(ipc) >= 1:
                            # Extract main section (first letter)
                            main_section = ipc[0].upper()
                            ipc_counter[main_section] += 1
                            total_classifications += 1
                elif isinstance(parsed_ipc, str) and len(parsed_ipc) >= 1:
                    # Handle single string IPC codes
                    main_section = parsed_ipc[0].upper()
                    ipc_counter[main_section] += 1
                    total_classifications += 1
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"Error parsing IPC: {ipc_data[:100]} - {e}")
    
    print(f"\nProcessed {patents_processed:,} patents")
    print(f"Total IPC classifications counted: {total_classifications:,}")
    print(f"Errors encountered: {error_count:,}")
    
    print(f"\nIPC Distribution by Section:")
    for section, count in ipc_counter.most_common():
        percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
        print(f"  {section}: {count:,} ({percentage:.1f}%)")
      # 5. Check for different IPC data formats
    print(f"\n=== IPC Technologies Data Format Analysis ===")
    cursor.execute("SELECT ipc_technologies FROM patents WHERE ipc_technologies IS NOT NULL AND ipc_technologies != '' LIMIT 100")
    format_samples = cursor.fetchall()
    
    json_array_count = 0
    json_string_count = 0
    plain_string_count = 0
    other_format_count = 0
    
    for (ipc_data,) in format_samples:
        try:
            parsed = json.loads(ipc_data)
            if isinstance(parsed, list):
                json_array_count += 1
            elif isinstance(parsed, str):
                json_string_count += 1
            else:
                other_format_count += 1
        except:
            plain_string_count += 1
    
    print(f"Sample of 100 IPC Technologies entries:")
    print(f"  JSON arrays: {json_array_count}")
    print(f"  JSON strings: {json_string_count}")
    print(f"  Plain strings: {plain_string_count}")
    print(f"  Other formats: {other_format_count}")
    
    conn.close()

if __name__ == "__main__":
    check_ipc_database()
