"""
Worker to evaluate records before fine-tuning.
Polls intune_db for status_eval_first='ready', computes metrics, updates columns.
"""
import os
import sys
import time
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client
from src.metrics.llm_eval import score_datapoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_int8(value):
    """Convert decimal metric to int8 (multiply by 10000, round)"""
    if value is None:
        return None
    return int(round(value * 10000))


def compute_metrics(record):
    """Compute metrics for a single record"""
    try:
        item = {
            "input": record.get("input", ""),
            "expected_output": record.get("expected_output", ""),
            "context": record.get("context", []),
            "actual_output": record.get("actual_output", "")
        }
        
        metrics = score_datapoint(item)
        
        # Round to 4 decimals
        return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error computing metrics for record {record.get('id')}: {e}")
        return None


def update_record(record_id, metrics):
    """Update record with metrics and mark as done"""
    try:
        supabase = get_supabase_client()
        
        update_data = {
            "answer_relevancy": to_int8(metrics.get("answer_relevancy")),
            "contextual_precision": to_int8(metrics.get("contextual_precision")),
            "contextual_recall": to_int8(metrics.get("contextual_recall")),
            "contextual_relevancy": to_int8(metrics.get("contextual_relevancy")),
            "faithfulness": to_int8(metrics.get("faithfulness")),
            "toxicity": to_int8(metrics.get("toxicity")),
            "hallucination_rate": to_int8(metrics.get("hallucination_rate")),
            "overall": to_int8(metrics.get("overall")),
            "status_eval_first": "done"
        }
        
        supabase.table("intune_db").update(update_data).eq("id", record_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error updating record {record_id}: {e}")
        return False


def main():
    logger.info("Starting first evaluation worker...")
    
    while True:
        try:
            supabase = get_supabase_client()
            
            # Fetch records with status_eval_first='ready'
            response = supabase.table("intune_db")\
                .select("*")\
                .eq("status_eval_first", "ready")\
                .limit(10)\
                .execute()
            
            records = response.data
            
            if records:
                logger.info(f"Found {len(records)} records to evaluate")
                
                for record in records:
                    record_id = record.get("id")
                    logger.info(f"Evaluating record {record_id}")
                    
                    metrics = compute_metrics(record)
                    
                    if metrics:
                        if update_record(record_id, metrics):
                            logger.info(f"✓ Updated record {record_id}")
                        else:
                            logger.error(f"✗ Failed to update record {record_id}")
                    else:
                        logger.error(f"✗ Failed to compute metrics for record {record_id}")
                
                time.sleep(5)  # Short pause between batches
            else:
                time.sleep(30)  # Wait longer if no records
                
        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
