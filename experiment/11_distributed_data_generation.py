"""
Step 11: Distributed Student Output Generation with Ray
=========================================================
Generates student outputs in PARALLEL across multiple machines for faster processing.

This is OPTIONAL - for speed only. The main pipeline (12_incremental_finetune.py)
can generate outputs locally on GPU.

Architecture:
  - Lenovo LOQ (HEAD) - coordinates + GPU generation
  - MacBook Pro (WORKER) - CPU generation via Ollama
  - MacBook Air (WORKER) - CPU generation via Ollama

Use Case:
  After training a checkpoint, you can generate student outputs faster
  by distributing across machines.

Usage:
    # Prepare Alpaca 50K data first
    python experiment/12_incremental_finetune.py --prepare-data
    
    # Generate for specific checkpoint (local mode):
    python experiment/11_distributed_data_generation.py --checkpoint 5 --local
    
    # Generate with Ray distributed:
    python experiment/11_distributed_data_generation.py --checkpoint 5 --distributed
    
    # Generate specific range of samples:
    python experiment/11_distributed_data_generation.py --checkpoint 5 --start 0 --end 1000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_SIZE = 5000
DATA_DIR = PROJECT_ROOT / "data" / "experiment"
MODELS_DIR = PROJECT_ROOT / "models" / "incremental_checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports" / "incremental_learning"

# Ollama config (for CPU workers)
OLLAMA_MODEL = "gemma3:1b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Worker hosts (update with your actual IPs)
WORKER_HOSTS = [
    "http://localhost:11434",      # Lenovo LOQ (local)
    "http://192.168.1.101:11434",  # MacBook Pro
    "http://192.168.1.102:11434",  # MacBook Air
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_eval_data(checkpoint: int, start: int = 0, end: Optional[int] = None) -> List[Dict]:
    """Load evaluation data from prepared Alpaca 50K"""
    data_file = DATA_DIR / "alpaca_50k_prepared.json"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Run: python experiment/12_incremental_finetune.py --prepare-data")
        sys.exit(1)
    
    with open(data_file, "r") as f:
        all_data = json.load(f)
    
    # Use last 5K for evaluation (consistent with main pipeline)
    eval_data = all_data[45000:]
    
    if end:
        eval_data = eval_data[start:end]
    else:
        eval_data = eval_data[start:]
    
    print(f"📊 Loaded {len(eval_data)} evaluation samples")
    return eval_data


# ============================================================================
# LOCAL GENERATION (GPU - Unsloth)
# ============================================================================

def generate_with_unsloth(
    eval_data: List[Dict],
    checkpoint: int
) -> List[Dict]:
    """Generate using Unsloth on GPU"""
    from unsloth import FastLanguageModel
    import torch
    from tqdm import tqdm
    
    # Load checkpoint model
    checkpoint_path = MODELS_DIR / f"checkpoint_{checkpoint}"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Train the checkpoint first with 12_incremental_finetune.py")
        return []
    
    print(f"🔄 Loading checkpoint {checkpoint} from {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # Prompt template
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    
    results = []
    
    for item in tqdm(eval_data, desc="Generating (GPU)"):
        prompt = prompt_template.format(
            instruction=item["instruction"],
            input=item["input"]
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start_time
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in full_output:
            student_output = full_output.split("### Response:")[-1].strip()
        else:
            student_output = full_output[len(prompt):].strip()
        
        results.append({
            "id": item["id"],
            "instruction": item["raw_instruction"],
            "context": item["context"],
            "input": item["input"],
            "teacher_output": item["teacher_output"],
            "student_output": student_output,
            "checkpoint": checkpoint,
            "latency": latency,
            "method": "unsloth_gpu"
        })
    
    return results


# ============================================================================
# LOCAL GENERATION (CPU - Ollama)
# ============================================================================

def generate_with_ollama(
    eval_data: List[Dict],
    checkpoint: int,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    model: str = OLLAMA_MODEL
) -> List[Dict]:
    """Generate using Ollama (CPU)"""
    import requests
    from tqdm import tqdm
    
    print(f"🔄 Generating with Ollama: {ollama_host}")
    print(f"   Model: {model}")
    
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    
    results = []
    
    for item in tqdm(eval_data, desc="Generating (Ollama)"):
        prompt = prompt_template.format(
            instruction=item["instruction"],
            input=item["input"]
        )
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 256
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            student_output = response.json().get("response", "").strip()
        except Exception as e:
            student_output = f"[ERROR: {str(e)}]"
        
        latency = time.time() - start_time
        
        results.append({
            "id": item["id"],
            "instruction": item["raw_instruction"],
            "context": item["context"],
            "input": item["input"],
            "teacher_output": item["teacher_output"],
            "student_output": student_output,
            "checkpoint": checkpoint,
            "latency": latency,
            "method": "ollama_cpu"
        })
    
    return results


# ============================================================================
# RAY DISTRIBUTED GENERATION
# ============================================================================

def generate_distributed(
    eval_data: List[Dict],
    checkpoint: int,
    num_workers: int = 3,
    worker_hosts: List[str] = None
) -> List[Dict]:
    """Generate using Ray distributed across multiple machines"""
    import ray
    
    if worker_hosts is None:
        worker_hosts = WORKER_HOSTS[:num_workers]
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
            print("✅ Connected to Ray cluster")
        except Exception:
            ray.init()
            print("✅ Started local Ray")
    
    # Define remote function
    @ray.remote
    def generate_batch(
        batch: List[Dict],
        ollama_host: str,
        model: str,
        checkpoint: int,
        worker_id: int
    ) -> List[Dict]:
        import requests
        import time
        
        prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
        
        results = []
        total = len(batch)
        
        for i, item in enumerate(batch):
            prompt = prompt_template.format(
                instruction=item["instruction"],
                input=item["input"]
            )
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 256}
                    },
                    timeout=120
                )
                response.raise_for_status()
                student_output = response.json().get("response", "").strip()
            except Exception as e:
                student_output = f"[ERROR: {str(e)}]"
            
            latency = time.time() - start_time
            
            results.append({
                "id": item["id"],
                "instruction": item["raw_instruction"],
                "context": item["context"],
                "input": item["input"],
                "teacher_output": item["teacher_output"],
                "student_output": student_output,
                "checkpoint": checkpoint,
                "latency": latency,
                "method": f"ray_worker_{worker_id}"
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Worker {worker_id}: {i + 1}/{total}")
        
        return results
    
    # Split data across workers
    chunk_size = len(eval_data) // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(eval_data)
        chunks.append(eval_data[start:end])
    
    print(f"\n🚀 Distributing {len(eval_data)} samples across {num_workers} workers:")
    for i, (chunk, host) in enumerate(zip(chunks, worker_hosts)):
        print(f"   Worker {i}: {len(chunk)} samples → {host}")
    
    # Launch tasks
    futures = []
    for i, (chunk, host) in enumerate(zip(chunks, worker_hosts)):
        future = generate_batch.remote(chunk, host, OLLAMA_MODEL, checkpoint, i)
        futures.append(future)
    
    # Collect results
    print("\n⏳ Waiting for workers...")
    all_results = []
    for future in futures:
        results = ray.get(future)
        all_results.extend(results)
        print(f"   ✓ Received {len(results)} results")
    
    print(f"\n✅ Total: {len(all_results)} outputs generated")
    return all_results


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_outputs(results: List[Dict], checkpoint: int, suffix: str = ""):
    """Save generated outputs"""
    outputs_dir = REPORTS_DIR / "student_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"checkpoint_{checkpoint}_outputs{suffix}.json"
    output_file = outputs_dir / filename
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Saved to {output_file}")
    
    # Stats
    successful = [r for r in results if not r["student_output"].startswith("[ERROR")]
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    
    print(f"\n📊 Generation Stats:")
    print(f"   Total: {len(results)}")
    print(f"   Successful: {len(successful)} ({100 * len(successful) / len(results):.1f}%)")
    print(f"   Avg Latency: {avg_latency:.2f}s")
    
    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Distributed Student Output Generation")
    parser.add_argument("--checkpoint", type=int, required=True, help="Checkpoint to use")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--local", action="store_true", help="Use local GPU (Unsloth)")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama (CPU)")
    parser.add_argument("--distributed", action="store_true", help="Use Ray distributed")
    parser.add_argument("--workers", type=int, default=3, help="Number of Ray workers")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DISTRIBUTED STUDENT OUTPUT GENERATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {'Local GPU' if args.local else 'Ollama' if args.ollama else 'Ray Distributed'}")
    print("=" * 60)
    
    # Load data
    eval_data = load_eval_data(args.checkpoint, args.start, args.end)
    
    # Generate
    if args.local:
        results = generate_with_unsloth(eval_data, args.checkpoint)
        suffix = "_gpu"
    elif args.ollama:
        results = generate_with_ollama(eval_data, args.checkpoint, args.ollama_host)
        suffix = "_ollama"
    elif args.distributed:
        results = generate_distributed(eval_data, args.checkpoint, args.workers)
        suffix = "_distributed"
    else:
        # Default to local GPU
        results = generate_with_unsloth(eval_data, args.checkpoint)
        suffix = "_gpu"
    
    # Save
    if results:
        save_outputs(results, args.checkpoint, suffix)
    else:
        print("❌ No results generated")


if __name__ == "__main__":
    main()
