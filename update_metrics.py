"""
Fetch records from Supabase, compute metrics, and update in place.
Also removes duplicate records (keeps most recent by created_at).
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Import the scoring function from llm_eval
import sys
matrics_path = os.path.join(os.path.dirname(__file__), 'matrics')
if matrics_path not in sys.path:
    sys.path.insert(0, matrics_path)
from matrics.llm_eval import score_datapoint

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_supabase_client() -> Client:
    """Create and return Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_records(supabase: Client) -> List[Dict[str, Any]]:
    """Fetch all records from Supabase"""
    print("Fetching records from Supabase...")
    
    # Fetch in batches to handle large datasets
    all_records = []
    batch_size = 10
    offset = 0
    
    while True:
        response = supabase.table("inference_results")\
            .select("*")\
            .order("created_at", desc=True)\
            .range(offset, offset + batch_size - 1)\
            .execute()
        
        records = response.data
        if not records:
            break
        
        all_records.extend(records)
        print(f"  Fetched {len(all_records)} records so far...", end="\r")
        
        if len(records) < batch_size:
            break
        
        offset += batch_size
    
    print(f"\n✓ Fetched {len(all_records)} total records")
    return all_records


def remove_duplicates(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[int]]:
    """
    Remove duplicate records by 'input' field.
    Keeps the most recent one (first in list since ordered by created_at DESC).
    Returns: (unique_records, ids_to_delete)
    """
    print("\nChecking for duplicate records...")
    
    seen_inputs = {}
    unique_records = []
    ids_to_delete = []
    
    for record in records:
        input_text = record.get("input", "")
        record_id = record.get("id")
        
        if input_text not in seen_inputs:
            # First occurrence - keep it
            seen_inputs[input_text] = record_id
            unique_records.append(record)
        else:
            # Duplicate - mark for deletion
            ids_to_delete.append(record_id)
    
    duplicates_count = len(ids_to_delete)
    print(f"✓ Found {duplicates_count} duplicate records")
    print(f"✓ Keeping {len(unique_records)} unique records")
    
    return unique_records, ids_to_delete


def delete_duplicate_records(supabase: Client, ids_to_delete: List[int]) -> None:
    """Delete duplicate records from Supabase"""
    if not ids_to_delete:
        print("✓ No duplicates to delete")
        return
    
    print(f"\nDeleting {len(ids_to_delete)} duplicate records...")
    
    # Delete in batches to avoid API limits
    batch_size = 100
    for i in range(0, len(ids_to_delete), batch_size):
        batch = ids_to_delete[i:i + batch_size]
        supabase.table("inference_results").delete().in_("id", batch).execute()
        print(f"  Deleted {min(i + batch_size, len(ids_to_delete))}/{len(ids_to_delete)} records...", end="\r")
    
    print(f"\n✓ Successfully deleted {len(ids_to_delete)} duplicate records")


def compute_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single record"""
    # Prepare item in the format expected by score_datapoint
    item = {
        "input": record.get("input", ""),
        "expected_output": record.get("expected_output", ""),
        "context": record.get("context", []),
        "actual_output": record.get("actual_output", "")
    }
    
    metrics = score_datapoint(item)
    
    # Round numeric metrics to 4 decimals
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rounded_metrics[key] = round(value, 4)
        else:
            rounded_metrics[key] = value
    
    return rounded_metrics


def to_int8(value):
    """Convert decimal metric to int8 (multiply by 10000, round)"""
    if value is None:
        return None
    return int(round(value * 10000))


def prepare_metrics_update(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metrics for update (convert to int8 format)"""
    return {
        "answer_relevancy": to_int8(metrics.get("answer_relevancy")),
        "contextual_precision": to_int8(metrics.get("contextual_precision")),
        "contextual_recall": to_int8(metrics.get("contextual_recall")),
        "contextual_relevancy": to_int8(metrics.get("contextual_relevancy")),
        "faithfulness": to_int8(metrics.get("faithfulness")),
        "toxicity": to_int8(metrics.get("toxicity")),
        "hallucination_rate": to_int8(metrics.get("hallucination_rate")),
        "overall": to_int8(metrics.get("overall"))
    }


def update_record_metrics(supabase: Client, record_id: int, metrics_data: Dict[str, Any]) -> None:
    """Update metrics for a single record"""
    supabase.table("inference_results")\
        .update(metrics_data)\
        .eq("id", record_id)\
        .execute()


def process_and_update_records(supabase: Client, records: List[Dict[str, Any]]) -> None:
    """Compute metrics and update each record"""
    print(f"\nProcessing and updating {len(records)} records...")
    
    for i, record in enumerate(records, 1):
        record_id = record.get("id")
        
        # Compute metrics
        metrics = compute_metrics(record)
        
        # Prepare update data
        metrics_data = prepare_metrics_update(metrics)
        
        # Update in Supabase
        update_record_metrics(supabase, record_id, metrics_data)
        
        print(f"  Updated {i}/{len(records)} records...", end="\r")
    
    print(f"\n✓ Successfully updated {len(records)} records with metrics!")


def main():
    """Main execution function"""
    print("=" * 60)
    print("SUPABASE METRICS UPDATE SCRIPT")
    print("=" * 60)
    
    # Create Supabase client
    supabase = get_supabase_client()
    
    # Step 1: Fetch all records
    all_records = fetch_all_records(supabase)
    
    if not all_records:
        print("\n⚠️  No records found in Supabase!")
        return
    
    # Step 2: Remove duplicates
    unique_records, ids_to_delete = remove_duplicates(all_records)
    
    # Step 3: Delete duplicate records
    if ids_to_delete:
        delete_duplicate_records(supabase, ids_to_delete)
    
    # Step 4: Compute metrics and update records
    process_and_update_records(supabase, unique_records)
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print(f"   - Total records fetched: {len(all_records)}")
    print(f"   - Duplicates removed: {len(ids_to_delete)}")
    print(f"   - Records updated with metrics: {len(unique_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
