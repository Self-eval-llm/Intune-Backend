"""
Evaluate fine-tuned model with BATCH INFERENCE for maximum speed.
Processes multiple prompts at once for ~2x speedup.

Features:
- Batch inference (process 4 prompts at once)
- Local checkpoint for resume support
- Progress bar with ETA
- 4-bit quantization for 8GB VRAM

Usage:
    python evaluate_finetuned_batch.py
    python evaluate_finetuned_batch.py --reset  # Start fresh
    python evaluate_finetuned_batch.py --batch-size 2  # Smaller batch if OOM
"""

# Windows compatibility: disable torch.compile/Triton before imports
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# Import from reorganized modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics.llm_eval import score_datapoint
from src.database.supabase_client import get_supabase_client

# Load environment variables
load_dotenv()

# Fine-tuned model path
FINETUNED_MODEL_PATH = os.path.join(project_root, 'models', 'gemma-finetuned', 'checkpoint-1233')

# Checkpoint file path
CHECKPOINT_FILE = os.path.join(project_root, 'reports', 'finetune_eval_checkpoint.json')
RESULTS_FILE = os.path.join(project_root, 'reports', 'finetune_eval_results.json')

# MANUAL SKIP: Records already evaluated
SKIP_FIRST_N_RECORDS = 346

# Batch size (adjust based on VRAM - 4 works for 8GB with 4-bit)
DEFAULT_BATCH_SIZE = 3


def load_checkpoint() -> Dict:
    """Load checkpoint from file"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_ids": [], "last_id": None, "start_time": None}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def reset_checkpoint():
    """Reset checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("✓ Checkpoint reset")
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print("✓ Results reset")


def load_results() -> List[Dict]:
    """Load accumulated results"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_results(results: List[Dict]):
    """Save accumulated results"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)


def fetch_records_with_metrics(supabase: Client, limit: int = None) -> List[Dict[str, Any]]:
    """Fetch records that have metrics from base model"""
    print(f"Fetching records with existing metrics...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        query = supabase.table("inference_results")\
            .select("*")\
            .not_.is_("answer_relevancy", "null")\
            .order("id")\
            .range(offset, offset + page_size - 1)
        
        if limit and offset >= limit:
            break
        
        response = query.execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    if limit:
        all_records = all_records[:limit]
    
    print(f"✓ Total fetched: {len(all_records)} records")
    return all_records


def load_finetuned_model():
    """Load fine-tuned model for inference"""
    from unsloth import FastLanguageModel
    
    print("\n" + "=" * 80)
    print("LOADING FINE-TUNED MODEL (4-bit for batch inference)")
    print("=" * 80)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,  # 4-bit for VRAM efficiency
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Enable left padding for batch generation
    tokenizer.padding_side = "left"
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print(f"✓ Loaded model from: {FINETUNED_MODEL_PATH}")
    print("✓ Inference mode enabled (4-bit quantized)")
    print("✓ Left padding enabled for batch inference")
    
    return model, tokenizer


def format_context(context: Any) -> str:
    """Format context into a string"""
    if not context:
        return ""
    if isinstance(context, list):
        return "\n".join(f"- {item}" for item in context if item)
    return str(context)


def build_prompt(record: Dict[str, Any]) -> str:
    """Build prompt for a single record"""
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
    return prompt


def generate_batch_outputs(model, tokenizer, records: List[Dict[str, Any]]) -> List[str]:
    """Generate outputs for a BATCH of records at once"""
    import torch
    
    # Build prompts for all records
    prompts = [build_prompt(record) for record in records]
    
    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    # Generate for batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode all outputs
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    # Extract responses
    responses = []
    for text in decoded:
        response = text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        responses.append(response)
    
    return responses


def compute_metrics_for_output(record: Dict[str, Any], output: str) -> Dict[str, Any]:
    """Compute metrics for generated output"""
    item = {
        "input": record.get("input", ""),
        "expected_output": record.get("expected_output", ""),
        "context": record.get("context", []),
        "actual_output": output
    }
    
    metrics = score_datapoint(item)
    
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rounded_metrics[key] = round(value, 4)
        else:
            rounded_metrics[key] = value
    
    return rounded_metrics


def to_int8(value):
    """Convert decimal metric to int8"""
    if value is None:
        return None
    return int(round(value * 10000))


def save_to_supabase(supabase: Client, record_id: int, output: str, metrics: Dict[str, Any]) -> None:
    """Save fine-tuned output and metrics to Supabase"""
    update_data = {
        "actual_output_tuned": output,
        "answer_relevancy_tuned": to_int8(metrics.get("answer_relevancy")),
        "contextual_precision_tuned": to_int8(metrics.get("contextual_precision")),
        "contextual_recall_tuned": to_int8(metrics.get("contextual_recall")),
        "contextual_relevancy_tuned": to_int8(metrics.get("contextual_relevancy")),
        "faithfulness_tuned": to_int8(metrics.get("faithfulness")),
        "toxicity_tuned": to_int8(metrics.get("toxicity")),
        "hallucination_rate_tuned": to_int8(metrics.get("hallucination_rate")),
        "overall_tuned": to_int8(metrics.get("overall"))
    }
    
    supabase.table("inference_results")\
        .update(update_data)\
        .eq("id", record_id)\
        .execute()


def evaluate_with_batch(model, tokenizer, supabase: Client, records: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    """Evaluate with BATCH inference - FAST!"""
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get("processed_ids", []))
    results = load_results()
    
    # Skip first N records (already done by old script)
    if SKIP_FIRST_N_RECORDS > 0 and len(processed_ids) == 0:
        print(f"\n⚠️  Skipping first {SKIP_FIRST_N_RECORDS} records (already evaluated)")
        records = records[SKIP_FIRST_N_RECORDS:]
    
    # Filter out already processed records
    remaining_records = [r for r in records if r.get("id") not in processed_ids]
    
    print("\n" + "=" * 80)
    print(f"BATCH EVALUATION (batch_size={batch_size})")
    print("=" * 80)
    print(f"Total records: {len(records)}")
    print(f"Skipped (old script): {SKIP_FIRST_N_RECORDS}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining: {len(remaining_records)}")
    
    if not remaining_records:
        print("✓ All records already processed!")
        return results
    
    # Set start time
    if not checkpoint.get("start_time"):
        checkpoint["start_time"] = datetime.now().isoformat()
        save_checkpoint(checkpoint)
    
    # Create batches
    batches = [remaining_records[i:i+batch_size] for i in range(0, len(remaining_records), batch_size)]
    
    start_time = time.time()
    total_processed = 0
    
    # Progress bar for batches
    pbar = tqdm(batches, desc="Batches", unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Generate outputs for entire batch
            outputs = generate_batch_outputs(model, tokenizer, batch)
            
            # Process each record in batch
            batch_improvements = []
            for record, output in zip(batch, outputs):
                record_id = record.get("id")
                
                # Compute metrics
                metrics = compute_metrics_for_output(record, output)
                
                # Save to Supabase
                save_to_supabase(supabase, record_id, output, metrics)
                
                # Get base metrics
                base_overall = record.get('overall', 0) / 10000.0 if record.get('overall') else 0
                tuned_overall = metrics.get('overall', 0)
                improvement = tuned_overall - base_overall
                
                batch_improvements.append(improvement)
                
                # Store result
                results.append({
                    'id': record_id,
                    'base_overall': base_overall,
                    'finetuned_overall': tuned_overall,
                    'improvement': improvement
                })
                
                # Update checkpoint
                processed_ids.add(record_id)
                total_processed += 1
            
            # Update checkpoint after each batch
            checkpoint["processed_ids"] = list(processed_ids)
            checkpoint["last_id"] = batch[-1].get("id")
            
            # Calculate stats
            avg_improvement = sum(batch_improvements) / len(batch_improvements)
            elapsed = time.time() - start_time
            rate = total_processed / elapsed
            remaining = len(remaining_records) - total_processed
            eta_seconds = remaining / rate if rate > 0 else 0
            
            pbar.set_postfix({
                'Done': total_processed,
                'Rate': f"{rate:.2f}/s",
                'Avg Δ': f"{avg_improvement:+.3f}",
                'ETA': str(timedelta(seconds=int(eta_seconds)))
            })
            
            # Save checkpoint every 5 batches
            if (batch_idx + 1) % 5 == 0:
                save_checkpoint(checkpoint)
                save_results(results)
                tqdm.write(f"  💾 Checkpoint saved at batch {batch_idx + 1}")
        
        except Exception as e:
            tqdm.write(f"  ⚠️ Error in batch {batch_idx}: {e}")
            # Save what we have
            save_checkpoint(checkpoint)
            save_results(results)
            continue
    
    # Final save
    save_checkpoint(checkpoint)
    save_results(results)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed {total_processed} records in {timedelta(seconds=int(elapsed))}")
    print(f"✓ Average rate: {total_processed/elapsed:.2f} records/second")
    print(f"✓ Checkpoint saved to: {CHECKPOINT_FILE}")
    
    return results


def generate_final_report(results: List[Dict], supabase: Client):
    """Generate comparison report from results"""
    if not results:
        print("No results to report")
        return
    
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    
    improvements = [r['improvement'] for r in results]
    base_scores = [r['base_overall'] for r in results]
    tuned_scores = [r['finetuned_overall'] for r in results]
    
    avg_base = sum(base_scores) / len(base_scores)
    avg_tuned = sum(tuned_scores) / len(tuned_scores)
    avg_improvement = sum(improvements) / len(improvements)
    
    improved_count = sum(1 for i in improvements if i > 0)
    declined_count = sum(1 for i in improvements if i < 0)
    
    print(f"\nTotal Records: {len(results)}")
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Avg Base Overall':<25} {avg_base:.4f}")
    print(f"{'Avg Finetuned Overall':<25} {avg_tuned:.4f}")
    print(f"{'Avg Improvement':<25} {avg_improvement:+.4f}")
    print(f"{'Improvement %':<25} {(avg_improvement/avg_base)*100:+.2f}%")
    print("-" * 40)
    print(f"{'Records Improved':<25} {improved_count} ({improved_count/len(results)*100:.1f}%)")
    print(f"{'Records Declined':<25} {declined_count} ({declined_count/len(results)*100:.1f}%)")
    
    # Save report
    report = {
        "total_records": len(results),
        "avg_base_overall": round(avg_base, 4),
        "avg_finetuned_overall": round(avg_tuned, 4),
        "avg_improvement": round(avg_improvement, 4),
        "improvement_percent": round((avg_improvement/avg_base)*100, 2),
        "records_improved": improved_count,
        "records_declined": declined_count,
        "timestamp": datetime.now().isoformat()
    }
    
    report_path = os.path.join(project_root, 'reports', 'finetune_final_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate fine-tuned model")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f"Batch size (default: {DEFAULT_BATCH_SIZE}, reduce if OOM)")
    args = parser.parse_args()
    
    if args.reset:
        reset_checkpoint()
        print("Starting fresh evaluation...")
    
    supabase = get_supabase_client()
    
    if args.report_only:
        results = load_results()
        generate_final_report(results, supabase)
        return
    
    records = fetch_records_with_metrics(supabase)
    
    if not records:
        print("No records found!")
        return
    
    model, tokenizer = load_finetuned_model()
    
    results = evaluate_with_batch(model, tokenizer, supabase, records, args.batch_size)
    
    generate_final_report(results, supabase)
    
    print("\n✅ Batch evaluation completed!")


if __name__ == "__main__":
    main()
