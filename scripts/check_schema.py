#!/usr/bin/env python3
"""
Fix database schema for teacher comparison evaluation.
Ensures all metric columns are INT type (not NUMERIC with limited precision).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def get_supabase_client() -> Client:
    """Create Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def verify_column_types(supabase: Client) -> dict:
    """Check current column types for metric columns"""
    print("Checking column types in modelComp table...")
    
    # Use raw SQL query to check column types
    result = supabase.rpc(
        "sql",
        {
            "query": """
            SELECT column_name, data_type, numeric_precision, numeric_scale
            FROM information_schema.columns
            WHERE table_name = 'modelComp'
            AND (column_name LIKE 'alpaca_%' OR column_name LIKE 'oss_%')
            ORDER BY column_name
            """
        }
    ).execute()
    
    columns = result.data if hasattr(result, 'data') else []
    
    print("\n📋 Current Column Types:")
    print("-" * 80)
    for col in columns:
        col_name = col.get('column_name', 'N/A')
        data_type = col.get('data_type', 'N/A')
        precision = col.get('numeric_precision', '')
        scale = col.get('numeric_scale', '')
        
        if precision:
            type_str = f"{data_type}({precision},{scale})"
        else:
            type_str = data_type
            
        status = "❌ NEEDS FIX" if "numeric" in data_type.lower() else "✓ OK"
        print(f"  {col_name:<30} {type_str:<20} {status}")
    
    return {col['column_name']: col['data_type'] for col in columns}


def print_fix_instructions():
    """Print instructions for fixing the schema"""
    print("\n" + "=" * 80)
    print("SCHEMA FIX INSTRUCTIONS")
    print("=" * 80)
    print("""
The database has NUMERIC(5,4) columns which can only store values from -9.9999 to 9.9999.
However, the evaluation metrics are converted to INT values (0-10000) using the to_int8() function.

SOLUTION:
1. Run the SQL migration script in Supabase SQL Editor:
   👉 Open: https://supabase.com → Your Project → SQL Editor
   👉 Copy and run: sql/fix_metric_column_types.sql

2. After running the migration, re-run the evaluation:
   python experiment/07_compare_teachers.py --dry-run

This will convert all NUMERIC(5,4) columns to INT columns that can store the full range
of metric values (0-10000).

For details, see: sql/fix_metric_column_types.sql
""")


def main():
    """Main execution"""
    try:
        supabase = get_supabase_client()
        column_types = verify_column_types(supabase)
        
        # Check if any columns need fixing
        needs_fix = any("numeric" in dtype.lower() for dtype in column_types.values())
        
        if needs_fix:
            print("\n" + "=" * 80)
            print("❌ SCHEMA ISSUE DETECTED")
            print("=" * 80)
            print("""
Some metric columns have type NUMERIC(5,4) which cannot store values > 9.9999.
The evaluation metrics are converted to 0-10000 range, causing overflow errors.
""")
            print_fix_instructions()
            return 1
        else:
            print("\n" + "=" * 80)
            print("✓ SCHEMA IS CORRECT")
            print("=" * 80)
            print("""
All metric columns are using INT type. You can proceed with evaluation.
Run: python experiment/07_compare_teachers.py
""")
            return 0
            
    except Exception as e:
        print(f"\n❌ Error checking schema: {e}")
        print_fix_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
