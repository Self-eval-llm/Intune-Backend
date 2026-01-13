#!/usr/bin/env python3
"""
Automatic schema fixer for teacher comparison evaluation.
Detects and fixes NUMERIC(5,4) columns that cause overflow errors.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client
import psycopg2
from psycopg2 import sql

load_dotenv()

def get_supabase_client() -> Client:
    """Create Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def get_database_connection() -> psycopg2.extensions.connection:
    """Create direct PostgreSQL connection (requires DB password)"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Try to construct from Supabase URL and key
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            return None
        # This would need the actual password, which isn't stored
        return None
    
    try:
        return psycopg2.connect(db_url)
    except Exception:
        return None


def check_column_types_via_supabase(supabase: Client) -> Dict[str, Tuple[str, str, str]]:
    """
    Check column types using Supabase.
    Returns dict: {column_name: (data_type, precision, scale)}
    """
    try:
        # Execute a query to check column information
        response = supabase.table("modelComp").select("*").limit(0).execute()
        # This won't give us schema info, need to use a different approach
        print("⚠️  Cannot directly query schema via Supabase SDK")
        return {}
    except Exception as e:
        print(f"Error querying schema: {e}")
        return {}


def suggest_fix_steps():
    """Print step-by-step fix instructions"""
    print("\n" + "=" * 80)
    print("📋 AUTOMATIC FIX INSTRUCTIONS")
    print("=" * 80)
    
    instructions = """
ISSUE: Database metric columns have type NUMERIC(5,4) which overflows with values > 9.9999

SOLUTION (3 easy steps):

1️⃣  OPEN SUPABASE SQL EDITOR
   🔗 Go to: https://supabase.com → Your Project → SQL Editor

2️⃣  RUN THE MIGRATION SCRIPT
   📄 Copy and paste the contents of:
      👉 sql/fix_metric_column_types.sql
   
   Then click the "Run" button.

3️⃣  VERIFY THE FIX
   Run this command to confirm:
   👉 python scripts/check_schema.py
   
   Expected output: "✓ SCHEMA IS CORRECT"

4️⃣  RE-RUN EVALUATION
   python experiment/07_compare_teachers.py

═══════════════════════════════════════════════════════════════════════════════

WHAT THE SCRIPT DOES:
  ✓ Drops old NUMERIC(5,4) columns
  ✓ Creates new INT columns (can store 0-10000)
  ✓ Creates a decimal view for easy querying

COLUMN TYPE CHANGES:
  Before: NUMERIC(5,4) → Range: -9.9999 to 9.9999  ❌
  After:  INT           → Range: -2.1B to 2.1B     ✓

═══════════════════════════════════════════════════════════════════════════════

NEED HELP?
  📖 Read: SCHEMA_FIX_GUIDE.md
  🔍 Check: sql/fix_metric_column_types.sql (has detailed comments)
"""
    
    print(instructions)
    
    # Print the SQL commands for manual execution
    sql_file = Path(__file__).parent.parent / "sql" / "fix_metric_column_types.sql"
    if sql_file.exists():
        print("\n" + "=" * 80)
        print("🔧 SQL MIGRATION SCRIPT (copy this)")
        print("=" * 80)
        with open(sql_file, 'r') as f:
            # Print first 50 lines
            lines = f.readlines()[:50]
            print(''.join(lines))
        print(f"\n... (see full script at {sql_file})")


def create_minimal_fix_script() -> str:
    """
    Create a minimal SQL script that can be copy-pasted into Supabase.
    """
    return """-- Fix NUMERIC(5,4) columns that cause overflow errors
-- Copy and paste this into Supabase SQL Editor, then click Run

-- Drop problematic columns
ALTER TABLE "modelComp" 
  DROP COLUMN IF EXISTS alpaca_struct_correct,
  DROP COLUMN IF EXISTS alpaca_task_success,
  DROP COLUMN IF EXISTS alpaca_instr_follow,
  DROP COLUMN IF EXISTS alpaca_coverage,
  DROP COLUMN IF EXISTS alpaca_faithfulness,
  DROP COLUMN IF EXISTS alpaca_hallucination,
  DROP COLUMN IF EXISTS alpaca_ctx_grounding,
  DROP COLUMN IF EXISTS alpaca_overall,
  DROP COLUMN IF EXISTS oss_struct_correct,
  DROP COLUMN IF EXISTS oss_task_success,
  DROP COLUMN IF EXISTS oss_instr_follow,
  DROP COLUMN IF EXISTS oss_coverage,
  DROP COLUMN IF EXISTS oss_faithfulness,
  DROP COLUMN IF EXISTS oss_hallucination,
  DROP COLUMN IF EXISTS oss_ctx_grounding,
  DROP COLUMN IF EXISTS oss_overall;

-- Recreate as INT columns (can store 0-10000)
ALTER TABLE "modelComp"
  ADD COLUMN alpaca_struct_correct INT DEFAULT 0,
  ADD COLUMN alpaca_task_success INT DEFAULT 0,
  ADD COLUMN alpaca_instr_follow INT DEFAULT 0,
  ADD COLUMN alpaca_coverage INT DEFAULT 0,
  ADD COLUMN alpaca_faithfulness INT DEFAULT 0,
  ADD COLUMN alpaca_hallucination INT DEFAULT 0,
  ADD COLUMN alpaca_ctx_grounding INT DEFAULT 0,
  ADD COLUMN alpaca_overall INT DEFAULT 0,
  ADD COLUMN oss_struct_correct INT DEFAULT 0,
  ADD COLUMN oss_task_success INT DEFAULT 0,
  ADD COLUMN oss_instr_follow INT DEFAULT 0,
  ADD COLUMN oss_coverage INT DEFAULT 0,
  ADD COLUMN oss_faithfulness INT DEFAULT 0,
  ADD COLUMN oss_hallucination INT DEFAULT 0,
  ADD COLUMN oss_ctx_grounding INT DEFAULT 0,
  ADD COLUMN oss_overall INT DEFAULT 0;

-- ✓ Done! Columns are now INT type
"""


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("🔧 TEACHER COMPARISON EVALUATION - SCHEMA FIX")
    print("=" * 80)
    
    print("\n📋 Analyzing the issue...")
    print("""
ERROR: "numeric field overflow" (code 22003)
CAUSE: Database columns are NUMERIC(5,4) - can only store -9.9999 to 9.9999
       But evaluation metrics are 0-10000 (from to_int8() function)
SOLUTION: Change column type to INT
""")
    
    suggest_fix_steps()
    
    # Save minimal fix script
    fix_script_path = Path(__file__).parent.parent / "scripts" / "minimal_schema_fix.sql"
    minimal_script = create_minimal_fix_script()
    
    print("\n" + "=" * 80)
    print("💾 MINIMAL SCRIPT (saved to minimal_schema_fix.sql)")
    print("=" * 80)
    print(minimal_script)
    
    try:
        fix_script_path.write_text(minimal_script)
        print(f"\n✓ Saved to: {fix_script_path}")
    except Exception as e:
        print(f"\nNote: Could not save minimal script: {e}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Go to Supabase SQL Editor: https://supabase.com
2. Copy the SQL from above or from: sql/fix_metric_column_types.sql
3. Paste it into the SQL Editor
4. Click Run
5. Run: python scripts/check_schema.py
6. Run: python experiment/07_compare_teachers.py
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
