"""
Worker to handle fine-tuning and post-finetune evaluation.
Checks for 5000 records with status_eval_first='done', runs finetune.py, 
then evaluates with fine-tuned model.
"""
import os
import sys
import time
import subprocess
import logging

# Windows compatibility
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client
from src.metrics.llm_eval import score_datapoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_int8(value):
    """Convert decimal metric to int8"""
    if value is None:
        return None
    return int(round(value * 10000))


def check_finetune_conditions():
    """Check if we have 2 records with status_eval_first='done' and empty status_eval_final"""
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .eq("status_eval_first", "done")\
            .is_("status_eval_final", "null")\
            .execute()
        
        count = response.count or 0
        logger.info(f"Found {count} records ready for fine-tuning")
        return count >= 2
    except Exception as e:
        logger.error(f"Error checking conditions: {e}")
        return False


def prepare_training_data():
    """Fetch training data from Supabase and create JSONL files for fine-tuning"""
    try:
        logger.info("Fetching training data from Supabase...")
        supabase = get_supabase_client()
        
        # Fetch records with status_eval_first='done' and status_eval_final=null
        response = supabase.table("intune_db")\
            .select("*")\
            .eq("status_eval_first", "done")\
            .is_("status_eval_final", "null")\
            .execute()
        
        records = response.data
        logger.info(f"Fetched {len(records)} records for training")
        
        if len(records) < 2:
            logger.warning("Not enough records for training")
            return False
        
        # Create training dataset in the format expected by finetune.py
        train_data = []
        for record in records:
            item = {
                "instruction": "Answer the following question accurately and concisely based on the provided information.",
                "input": record.get("input", ""),
                "output": record.get("expected_output", "") or record.get("actual_output", "")
            }
            train_data.append(item)
        
        # Split into train/val (80/20)
        split_idx = int(len(train_data) * 0.8)
        train_set = train_data[:split_idx]
        val_set = train_data[split_idx:]
        
        # Ensure data directories exist
        data_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)
        
        # Write JSONL files
        import json
        train_file = os.path.join(data_dir, 'train_dataset.jsonl')
        val_file = os.path.join(data_dir, 'val_dataset.jsonl')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_set:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_set:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ Created training dataset: {len(train_set)} train, {len(val_set)} val")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return False


def run_finetune():
    """Execute finetune.py"""
    try:
        finetune_script = os.path.join(project_root, 'src', 'training', 'finetune.py')
        
        if not os.path.exists(finetune_script):
            logger.error(f"Finetune script not found: {finetune_script}")
            return False
        
        logger.info("Starting fine-tuning process...")
        
        process = subprocess.Popen(
            [sys.executable, finetune_script],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"Finetune: {line.strip()}")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Fine-tuning completed successfully")
            return True
        else:
            logger.error(f"❌ Fine-tuning failed with code {return_code}")
            return False
    except Exception as e:
        logger.error(f"Error running finetune: {e}")
        return False


def load_finetuned_model():
    """Load fine-tuned model for inference"""
    try:
        from unsloth import FastLanguageModel
        
        model_path = os.path.join(project_root, 'models', 'gemma-finetuned-merged')
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False
        )
        
        FastLanguageModel.for_inference(model)
        logger.info(f"✓ Loaded fine-tuned model from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def format_context(context):
    """Format context into string"""
    if not context:
        return ""
    if isinstance(context, list):
        return "\n".join(f"- {item}" for item in context if item)
    return str(context)


def generate_with_finetuned(model, tokenizer, record):
    """Generate output using fine-tuned model"""
    try:
        question = record.get("input", "")
        context = format_context(record.get("context"))
        
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
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            use_cache=True
        )
        
        response = tokenizer.batch_decode(outputs)[0]
        response = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        return response
    except Exception as e:
        logger.error(f"Error generating output: {e}")
        return None


def compute_finetuned_metrics(record, output):
    """Compute metrics for fine-tuned output"""
    try:
        item = {
            "input": record.get("input", ""),
            "expected_output": record.get("expected_output", ""),
            "context": record.get("context", []),
            "actual_output": output
        }
        
        metrics = score_datapoint(item)
        return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return None


def update_finetuned_record(record_id, output, metrics):
    """Update record with fine-tuned output and metrics"""
    try:
        supabase = get_supabase_client()
        
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
            "status_eval_final": "done"
        }
        
        supabase.table("intune_db").update(update_data).eq("id", record_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error updating record {record_id}: {e}")
        return False


def evaluate_with_finetuned_model():
    """Evaluate all pending records with fine-tuned model"""
    logger.info("Starting post-finetune evaluation...")
    
    model, tokenizer = load_finetuned_model()
    if model is None or tokenizer is None:
        logger.error("Cannot load model for evaluation")
        return False
    
    try:
        supabase = get_supabase_client()
        
        while True:
            # Fetch records needing final evaluation
            response = supabase.table("intune_db")\
                .select("*")\
                .eq("status_eval_first", "done")\
                .is_("status_eval_final", "null")\
                .limit(5)\
                .execute()
            
            records = response.data
            
            if not records:
                logger.info("All records evaluated with fine-tuned model")
                break
            
            logger.info(f"Evaluating {len(records)} records with fine-tuned model")
            
            for record in records:
                record_id = record.get("id")
                logger.info(f"Processing record {record_id}")
                
                output = generate_with_finetuned(model, tokenizer, record)
                
                if output:
                    metrics = compute_finetuned_metrics(record, output)
                    
                    if metrics:
                        if update_finetuned_record(record_id, output, metrics):
                            logger.info(f"✓ Updated record {record_id}")
                        else:
                            logger.error(f"✗ Failed to update record {record_id}")
                    else:
                        logger.error(f"✗ Failed to compute metrics for record {record_id}")
                else:
                    logger.error(f"✗ Failed to generate output for record {record_id}")
            
            time.sleep(2)  # Small pause between batches
        
        return True
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return False


def main():
    logger.info("Starting fine-tune worker...")
    
    finetune_done = False
    
    while True:
        try:
            if not finetune_done:
                # Check if we should run fine-tuning
                if check_finetune_conditions():
                    logger.info("🎯 Conditions met! Preparing training data...")
                    
                    # Prepare training data from Supabase
                    if not prepare_training_data():
                        logger.error("Failed to prepare training data, retrying...")
                        time.sleep(300)
                        continue
                    
                    logger.info("Starting fine-tuning...")
                    
                    if run_finetune():
                        logger.info("✅ Fine-tuning completed")
                        finetune_done = True
                    else:
                        logger.error("❌ Fine-tuning failed, will retry")
                        time.sleep(600)  # Wait 10 minutes before retry
                else:
                    logger.info("⏳ Waiting for 2 evaluated records...")
                    time.sleep(300)  # Check every 5 minutes
            else:
                # Fine-tuning done, now evaluate with fine-tuned model
                logger.info("Starting final evaluation with fine-tuned model...")
                
                if evaluate_with_finetuned_model():
                    logger.info("🎉 All evaluations complete!")
                    break  # Exit after completing all evaluations
                else:
                    logger.error("Evaluation incomplete, retrying...")
                    time.sleep(60)
                    
        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            time.sleep(60)
    
    logger.info("Worker finished")


if __name__ == "__main__":
    main()
