"""
Distributed Worker for Output Generation
=========================================
Runs on MacBook Pro or any remote machine.
Pulls inference jobs from Supabase and generates outputs using finetuned model.
Tracks progress and prevents collisions with other workers.

Usage:
    # On MacBook:
    python 13_distributed_worker.py --checkpoint 1 --worker-id mac-1
    
    # On RTX Windows (optional parallel worker):
    python 13_distributed_worker.py --checkpoint 1 --worker-id rtx-1
"""

import os
import sys
import time
import argparse
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

# Fix Unicode/Emoji encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure logs are flushed immediately (prevents "looks stuck" behavior)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

load_dotenv()


# MEMORY OPTIMIZATION: Limit CPU usage for multiprocessing
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
# Limit datasets library to 2 workers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# CRITICAL: Disable torch.compile() which fails on Windows
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['DISABLE_UNSLOTH_AUTOIMPORT'] = '1'
# Windows-specific fixes for torch inductor
os.environ['INDUCTOR_NOBUILD'] = '1'
os.environ['TORCHINDUCTOR_CPP_WRAPPER'] = '0'
# Disable torch.compile/dynamo entirely on Windows (Triton incompatibility)
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Config (match with main script)
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512

def get_supabase():
    return create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def generate_output(model, tokenizer, instruction, context=""):
    """Generate output with model"""
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - start_time) * 1000
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response, latency

def get_progress_stats(supabase, checkpoint):
    """Get progress statistics for all workers"""
    total_result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', checkpoint)\
        .limit(1)\
        .execute()

    completed_result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', checkpoint)\
        .eq('status', 'score_tuned')\
        .limit(1)\
        .execute()

    stats = {
        'total': total_result.count or 0,
        'completed': completed_result.count or 0,
        'pending': 0,
        'workers': {}
    }

    page_size = 1000
    offset = 0
    while True:
        rows_result = supabase.table('modelcomp_50k')\
            .select('tuned_worker')\
            .eq('checkpoint', checkpoint)\
            .eq('status', 'score_tuned')\
            .range(offset, offset + page_size - 1)\
            .execute()

        rows = rows_result.data or []
        if not rows:
            break

        for row in rows:
            worker = row.get('tuned_worker') or 'unknown'
            stats['workers'][worker] = stats['workers'].get(worker, 0) + 1

        if len(rows) < page_size:
            break
        offset += page_size

    stats['pending'] = max(0, stats['total'] - stats['completed'])
    return stats

def get_other_workers(supabase, checkpoint, exclude_worker_id):
    """Get list of active workers"""
    stats = get_progress_stats(supabase, checkpoint)
    return [w for w in stats['workers'].keys() if w != 'unknown' and w != exclude_worker_id]

def estimate_completion_time(checkpoint, worker_id, batch_num, avg_time_per_record, pending_records):
    """Estimate time to completion"""
    if pending_records == 0:
        return None
    
    # Assume 2 workers averaging processing
    effective_records_per_sec = (2 / avg_time_per_record) if avg_time_per_record > 0 else 0
    remaining_seconds = pending_records / effective_records_per_sec if effective_records_per_sec > 0 else 0
    
    completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
    return completion_time, remaining_seconds

def worker_loop(checkpoint, worker_id, batch_size=10):
    """Main worker loop - continuously fetch and process batches"""
    print(f"\n{'='*70}")
    print(f"🚀 DISTRIBUTED WORKER: {worker_id}")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Batch size: {batch_size}")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    print(f"[1/4] Connecting to Supabase...")
    supabase = get_supabase()
    print(f"✅ Supabase connected")
    
    # Load model once
    model_path = f"models/gemma-ckpt{checkpoint}-lora"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return False
    
    print(f"\n[2/5] Importing Unsloth...")
    from unsloth import FastLanguageModel
    print(f"✅ Unsloth imported")

    print(f"[3/5] Loading model from {model_path}...")
    print(f"      (This may take 30-60 seconds on first load)")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        print(f"✅ Model weights loaded")
        
        print(f"[4/5] Preparing for inference...")
        FastLanguageModel.for_inference(model)
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"✅ Model inference ready on: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    finally:
        # Ensure GPU memory is cleared on any error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"[5/5] Starting batch processing loop...\n")
    
    total_generated = 0
    batch_count = 0
    record_times = []
    
    try:
        while True:
            # Get progress from all workers
            stats = get_progress_stats(supabase, checkpoint)
            other_workers = get_other_workers(supabase, checkpoint, worker_id)
            
            # Print progress dashboard
            print(f"\n{'─'*70}")
            print(f"PROGRESS DASHBOARD - Batch {batch_count + 1}")
            print(f"{'─'*70}")
            print(f"Total records: {stats['total']}")
            print(f"Completed: {stats['completed']} | Pending: {stats['pending']}")
            print(f"Progress: {(stats['completed']/stats['total']*100):.1f}%")
            
            if stats['workers']:
                print(f"\nWorker contributions:")
                for w, count in stats['workers'].items():
                    print(f"  {w}: {count} records")
            
            if other_workers:
                print(f"\nActive workers: {', '.join(other_workers)}")
            
            # Estimate completion
            if record_times:
                avg_time = sum(record_times[-10:]) / len(record_times[-10:])
                completion_info = estimate_completion_time(
                    checkpoint, worker_id, batch_count, avg_time, stats['pending']
                )
                if completion_info:
                    completion_time, remaining_sec = completion_info
                    print(f"\nEstimated completion: {completion_time.strftime('%H:%M:%S')}")
                    print(f"Time remaining: {int(remaining_sec) // 60}m {int(remaining_sec) % 60}s")
            
            print(f"{'─'*70}\n")
            
            # Fetch batch with collision avoidance
            # Only fetch records not assigned to other workers
            result = supabase.table('modelcomp_50k')\
                .select('*')\
                .eq('checkpoint', checkpoint)\
                .eq('status', 'output_tuned')\
                .is_('processing_worker', 'null')\
                .limit(batch_size)\
                .execute()
            
            batch = result.data if result.data else []
            
            if not batch:
                print(f"No unprocessed records. Checking if all done...")
                time.sleep(5)
                
                # Check if all completed
                final_stats = get_progress_stats(supabase, checkpoint)
                if final_stats['pending'] == 0:
                    print(f"\nAll records completed!")
                    break
                else:
                    print(f"Still {final_stats['pending']} pending. Retrying in 30s...")
                    time.sleep(30)
                    continue
            
            print(f"Processing batch {batch_count + 1}: {len(batch)} records")
            
            for idx, item in enumerate(batch):
                try:
                    # Mark as processing to prevent collision
                    supabase.table('modelcomp_50k').update({
                        'processing_worker': worker_id,
                        'processing_at': datetime.utcnow().isoformat()
                    }).eq('id', item['id']).execute()
                    
                    # Generate
                    gen_start = time.time()
                    output, latency = generate_output(
                        model, tokenizer,
                        item['input'],
                        item.get('context', '')
                    )
                    gen_time = time.time() - gen_start
                    record_times.append(gen_time)
                    
                    # Update results
                    supabase.table('modelcomp_50k').update({
                        'student_output_tuned': output[:5000],
                        'latency_tuned': round(latency, 3),
                        'status': 'score_tuned',
                        'tuned_worker': worker_id,
                        'tuned_at': datetime.utcnow().isoformat(),
                        'processing_worker': None
                    }).eq('id', item['id']).execute()
                    
                    total_generated += 1
                    
                    # Show progress within batch
                    avg_rec_time = sum(record_times) / len(record_times)
                    print(f"  [{idx+1}/{len(batch)}] Gen time: {gen_time:.2f}s | Avg: {avg_rec_time:.2f}s")
                    
                except Exception as e:
                    # Clear processing_worker on error
                    try:
                        supabase.table('modelcomp_50k').update({
                            'processing_worker': None
                        }).eq('id', item['id']).execute()
                    except:
                        pass
                    print(f"  ERROR on record {item['id']}: {e}")
                    continue
            
            batch_count += 1
            print(f"\nBatch {batch_count} complete. Total generated: {total_generated}")
            
    except KeyboardInterrupt:
        print(f"\n\nWorker interrupted.")
        print(f"Total generated by {worker_id}: {total_generated}")
    except Exception as e:
        print(f"\nFatal error: {e}")
        return False
    finally:
        # SECURITY FIX: Always cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model, tokenizer
    
    return True

def main():
    print("\n" + "="*70)
    print("🚀 DISTRIBUTED WORKER STARTUP")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__ if hasattr(torch, '__version__') else 'unknown'}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    parser = argparse.ArgumentParser(description='Distributed Worker for Inference')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint number')
    parser.add_argument('--worker-id', type=str, required=True, help='Worker ID (e.g., mac-1, rtx-1)')
    parser.add_argument('--batch-size', type=int, default=10, help='Records per batch')
    args = parser.parse_args()
    
    success = worker_loop(args.checkpoint, args.worker_id, args.batch_size)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
