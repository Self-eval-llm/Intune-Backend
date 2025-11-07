"""
Evaluation worker module for background evaluation functionality.
Contains worker logic and evaluation functions for first and final evaluations.
"""
import os
import sys
import threading
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import evaluation modules
from src.database.supabase_client import get_supabase_client
from src.metrics.llm_eval import score_datapoint

# Configure logging
logger = logging.getLogger(__name__)

# Global worker status
workers_running = {
    "first_eval": False,
    "final_eval": False
}

# Worker control
worker_threads = {
    "first_eval": None,
    "final_eval": None
}

# Startup time for uptime calculation
startup_time = datetime.now()


def to_int8(value):
    """Convert decimal metric to int8 (multiply by 10000, round)"""
    if value is None:
        return None
    return int(round(value * 10000))


def compute_metrics_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single record using existing llm_eval functionality"""
    try:
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
    except Exception as e:
        logger.error(f"Error computing metrics for record {record.get('id')}: {e}")
        return {}


def update_base_metrics_in_supabase(record_id: int, metrics: Dict[str, Any]) -> bool:
    """Update base model metrics in Supabase"""
    try:
        supabase = get_supabase_client()
        
        # Prepare metrics update (convert to int8 format)
        metrics_data = {
            "answer_relevancy": to_int8(metrics.get("answer_relevancy")),
            "contextual_precision": to_int8(metrics.get("contextual_precision")),
            "contextual_recall": to_int8(metrics.get("contextual_recall")),
            "contextual_relevancy": to_int8(metrics.get("contextual_relevancy")),
            "faithfulness": to_int8(metrics.get("faithfulness")),
            "toxicity": to_int8(metrics.get("toxicity")),
            "hallucination_rate": to_int8(metrics.get("hallucination_rate")),
            "overall": to_int8(metrics.get("overall")),
            "status_eval_first": True  # Mark as completed
        }
        
        supabase.table("intune_db")\
            .update(metrics_data)\
            .eq("id", record_id)\
            .execute()
        
        return True
    except Exception as e:
        logger.error(f"Error updating base metrics for record {record_id}: {e}")
        return False


def load_finetuned_model_and_generate():
    """Load fine-tuned model (simplified version for background worker)"""
    try:
        # Import here to avoid loading on startup
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        
        from unsloth import FastLanguageModel
        
        finetuned_model_path = os.path.join(project_root, 'models', 'gemma-finetuned-merged')
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=finetuned_model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        return None, None


def format_context(context: Any) -> str:
    """Format context into a string"""
    if not context:
        return ""
    
    if isinstance(context, list):
        return "\n".join(f"- {item}" for item in context if item)
    
    return str(context)


def generate_output_with_finetuned(model, tokenizer, record: Dict[str, Any]) -> str:
    """Generate output using fine-tuned model"""
    try:
        question = record.get("input", "")
        context = format_context(record.get("context"))
        
        # Create prompt in the same format as training
        instruction = "Answer the following question accurately and concisely based on the provided information."
        
        if context:
            input_text = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            input_text = f"Question: {question}"
        
        prompt = f"""<bos><start_of_turn>user
{instruction}

{input_text}<end_of_turn>
<start_of_turn>model
"""
        
        # Generate
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            use_cache=True
        )
        
        # Extract response
        response = tokenizer.batch_decode(outputs)[0]
        response = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        
        return response
    except Exception as e:
        logger.error(f"Error generating output with fine-tuned model: {e}")
        return ""


def compute_finetuned_metrics_for_record(record: Dict[str, Any], finetuned_output: str) -> Dict[str, Any]:
    """Compute metrics for fine-tuned output"""
    try:
        item = {
            "input": record.get("input", ""),
            "expected_output": record.get("expected_output", ""),
            "context": record.get("context", []),
            "actual_output": finetuned_output
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
    except Exception as e:
        logger.error(f"Error computing fine-tuned metrics for record {record.get('id')}: {e}")
        return {}


def update_finetuned_metrics_in_supabase(record_id: int, output: str, metrics: Dict[str, Any]) -> bool:
    """Update fine-tuned model metrics in Supabase"""
    try:
        supabase = get_supabase_client()
        
        # Prepare update data for *_tuned columns
        update_data = {
            "actual_output_tuned": output,
            "answer_relevancy_tuned": to_int8(metrics.get("answer_relevancy")),
            "contextual_precision_tuned": to_int8(metrics.get("contextual_precision")),
            "contextual_recall_tuned": to_int8(metrics.get("contextual_recall")),
            "contextual_relevancy_tuned": to_int8(metrics.get("contextual_relevancy")),
            "faithfulness_tuned": to_int8(metrics.get("faithfulness")),
            "toxicity_tuned": to_int8(metrics.get("toxicity")),
            "hallucination_rate_tuned": to_int8(metrics.get("hallucination_rate")),
            "overall_tuned": to_int8(metrics.get("overall")),
            "status_eval_final": True  # Mark as completed
        }
        
        supabase.table("intune_db")\
            .update(update_data)\
            .eq("id", record_id)\
            .execute()
        
        return True
    except Exception as e:
        logger.error(f"Error updating fine-tuned metrics for record {record_id}: {e}")
        return False


def first_evaluation_worker():
    """
    Background worker that monitors status_eval_first column and runs base model evaluation.
    Equivalent to running update_metrics.py on specific records.
    """
    logger.info("Starting first evaluation worker...")
    
    while workers_running["first_eval"]:
        try:
            supabase = get_supabase_client()
            
            # Fetch records where status_eval_first is False
            response = supabase.table("intune_db")\
                .select("*")\
                .eq("status_eval_first", False)\
                .limit(10)\
                .execute()  # Process 10 at a time to avoid overwhelming
            
            records = response.data
            
            if records:
                logger.info(f"Found {len(records)} records needing first evaluation")
                
                for record in records:
                    if not workers_running["first_eval"]:
                        break
                        
                    record_id = record.get("id")
                    logger.info(f"Processing first evaluation for record ID: {record_id}")
                    
                    # Compute metrics using existing functionality
                    metrics = compute_metrics_for_record(record)
                    
                    if metrics:
                        # Update in Supabase
                        if update_base_metrics_in_supabase(record_id, metrics):
                            logger.info(f"✓ Updated base metrics for record {record_id}")
                        else:
                            logger.error(f"✗ Failed to update base metrics for record {record_id}")
                    else:
                        logger.error(f"✗ Failed to compute metrics for record {record_id}")
            else:
                # No records to process, wait longer
                time.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in first evaluation worker: {e}")
            time.sleep(60)  # Wait before retrying on error
        
        # Short sleep between checks
        time.sleep(5)
    
    logger.info("First evaluation worker stopped")


def final_evaluation_worker():
    """
    Background worker that monitors status_eval_final column and runs fine-tuned model evaluation.
    Equivalent to running evaluate_finetuned.py on specific records.
    """
    logger.info("Starting final evaluation worker...")
    
    # Load fine-tuned model once (expensive operation)
    model, tokenizer = load_finetuned_model_and_generate()
    
    if model is None or tokenizer is None:
        logger.error("Failed to load fine-tuned model. Final evaluation worker cannot start.")
        workers_running["final_eval"] = False
        return
    
    logger.info("Fine-tuned model loaded successfully")
    
    while workers_running["final_eval"]:
        try:
            supabase = get_supabase_client()
            
            # Fetch records where status_eval_final is False
            # AND status_eval_first is True (base evaluation completed)
            response = supabase.table("intune_db")\
                .select("*")\
                .eq("status_eval_final", False)\
                .eq("status_eval_first", True)\
                .limit(5)\
                .execute()  # Process 5 at a time (model inference is slower)
            
            records = response.data
            
            if records:
                logger.info(f"Found {len(records)} records needing final evaluation")
                
                for record in records:
                    if not workers_running["final_eval"]:
                        break
                        
                    record_id = record.get("id")
                    logger.info(f"Processing final evaluation for record ID: {record_id}")
                    
                    # Generate output with fine-tuned model
                    finetuned_output = generate_output_with_finetuned(model, tokenizer, record)
                    
                    if finetuned_output:
                        # Compute metrics for fine-tuned output
                        metrics = compute_finetuned_metrics_for_record(record, finetuned_output)
                        
                        if metrics:
                            # Update in Supabase
                            if update_finetuned_metrics_in_supabase(record_id, finetuned_output, metrics):
                                logger.info(f"✓ Updated fine-tuned metrics for record {record_id}")
                            else:
                                logger.error(f"✗ Failed to update fine-tuned metrics for record {record_id}")
                        else:
                            logger.error(f"✗ Failed to compute fine-tuned metrics for record {record_id}")
                    else:
                        logger.error(f"✗ Failed to generate fine-tuned output for record {record_id}")
            else:
                # No records to process, wait longer
                time.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in final evaluation worker: {e}")
            time.sleep(120)  # Wait longer before retrying on error
        
        # Short sleep between checks
        time.sleep(10)
    
    logger.info("Final evaluation worker stopped")


# Worker control functions
def get_worker_status() -> dict:
    """Get current status of evaluation workers"""
    global workers_running, startup_time
    
    uptime = int((datetime.now() - startup_time).total_seconds())
    
    return {
        "first_eval_worker": workers_running["first_eval"],
        "final_eval_worker": workers_running["final_eval"],
        "uptime_seconds": uptime
    }


def start_first_eval_worker() -> dict:
    """Start the first evaluation background worker"""
    global workers_running, worker_threads
    
    if workers_running["first_eval"]:
        return {"success": False, "message": "First evaluation worker is already running"}
    
    workers_running["first_eval"] = True
    worker_threads["first_eval"] = threading.Thread(target=first_evaluation_worker, daemon=True)
    worker_threads["first_eval"].start()
    
    logger.info("First evaluation worker started")
    return {"success": True, "message": "First evaluation worker started successfully"}


def stop_first_eval_worker() -> dict:
    """Stop the first evaluation background worker"""
    global workers_running, worker_threads
    
    if not workers_running["first_eval"]:
        return {"success": False, "message": "First evaluation worker is not running"}
    
    workers_running["first_eval"] = False
    
    # Wait for thread to finish
    if worker_threads["first_eval"] and worker_threads["first_eval"].is_alive():
        worker_threads["first_eval"].join(timeout=10)
    
    logger.info("First evaluation worker stopped")
    return {"success": True, "message": "First evaluation worker stopped successfully"}


def start_final_eval_worker() -> dict:
    """Start the final evaluation background worker"""
    global workers_running, worker_threads
    
    if workers_running["final_eval"]:
        return {"success": False, "message": "Final evaluation worker is already running"}
    
    workers_running["final_eval"] = True
    worker_threads["final_eval"] = threading.Thread(target=final_evaluation_worker, daemon=True)
    worker_threads["final_eval"].start()
    
    logger.info("Final evaluation worker started")
    return {"success": True, "message": "Final evaluation worker started successfully"}


def stop_final_eval_worker() -> dict:
    """Stop the final evaluation background worker"""
    global workers_running, worker_threads
    
    if not workers_running["final_eval"]:
        return {"success": False, "message": "Final evaluation worker is not running"}
    
    workers_running["final_eval"] = False
    
    # Wait for thread to finish
    if worker_threads["final_eval"] and worker_threads["final_eval"].is_alive():
        worker_threads["final_eval"].join(timeout=30)  # Longer timeout for model cleanup
    
    logger.info("Final evaluation worker stopped")
    return {"success": True, "message": "Final evaluation worker stopped successfully"}


def start_all_workers() -> dict:
    """Start both evaluation workers"""
    global workers_running, worker_threads
    
    results = []
    
    # Start first evaluation worker
    if not workers_running["first_eval"]:
        workers_running["first_eval"] = True
        worker_threads["first_eval"] = threading.Thread(target=first_evaluation_worker, daemon=True)
        worker_threads["first_eval"].start()
        results.append("First evaluation worker started")
    else:
        results.append("First evaluation worker already running")
    
    # Start final evaluation worker
    if not workers_running["final_eval"]:
        workers_running["final_eval"] = True
        worker_threads["final_eval"] = threading.Thread(target=final_evaluation_worker, daemon=True)
        worker_threads["final_eval"].start()
        results.append("Final evaluation worker started")
    else:
        results.append("Final evaluation worker already running")
    
    logger.info("All evaluation workers started")
    return {"success": True, "message": "All workers started", "details": results}


def stop_all_workers() -> dict:
    """Stop both evaluation workers"""
    global workers_running, worker_threads
    
    results = []
    
    # Stop first evaluation worker
    if workers_running["first_eval"]:
        workers_running["first_eval"] = False
        if worker_threads["first_eval"] and worker_threads["first_eval"].is_alive():
            worker_threads["first_eval"].join(timeout=10)
        results.append("First evaluation worker stopped")
    else:
        results.append("First evaluation worker was not running")
    
    # Stop final evaluation worker
    if workers_running["final_eval"]:
        workers_running["final_eval"] = False
        if worker_threads["final_eval"] and worker_threads["final_eval"].is_alive():
            worker_threads["final_eval"].join(timeout=30)
        results.append("Final evaluation worker stopped")
    else:
        results.append("Final evaluation worker was not running")
    
    logger.info("All evaluation workers stopped")
    return {"success": True, "message": "All workers stopped", "details": results}


def get_pending_evaluations() -> dict:
    """Get count of pending evaluations"""
    try:
        supabase = get_supabase_client()
        
        # Count pending first evaluations
        first_eval_response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .eq("status_eval_first", False)\
            .execute()
        
        first_eval_count = first_eval_response.count
        
        # Count pending final evaluations (requires first eval to be completed)
        final_eval_response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .eq("status_eval_final", False)\
            .eq("status_eval_first", True)\
            .execute()
        
        final_eval_count = final_eval_response.count
        
        return {
            "pending_first_evaluations": first_eval_count,
            "pending_final_evaluations": final_eval_count,
            "total_pending": first_eval_count + final_eval_count
        }
        
    except Exception as e:
        logger.error(f"Error getting pending evaluations: {e}")
        return {
            "pending_first_evaluations": 0,
            "pending_final_evaluations": 0,
            "total_pending": 0,
            "error": str(e)
        }

