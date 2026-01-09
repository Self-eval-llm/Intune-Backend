"""
Step 06b: Generate Tuned-OSS20B Outputs for Teacher Comparison
==============================================================
Generates outputs from the Gemma model fine-tuned on OSS-20B synthetic data.
Updates `tuned_oss20b` column in modelComp table.

⚠️ Runs on MacBook Pro (M1/M2/M3)

Features:
- BATCH INFERENCE for 3-5x faster generation
- Supports Transformers, MLX, and Ollama backends
- Resume-safe (only processes NULL records)

Prerequisites:
- Fine-tuned model at: models/gemma-oss-teacher/
- Or merged model for MLX/Ollama

Usage:
    python experiment/06b_generate_tuned_oss.py
    python experiment/06b_generate_tuned_oss.py --limit 100  # Test subset
    python experiment/06b_generate_tuned_oss.py --batch-size 8  # Batch inference
    python experiment/06b_generate_tuned_oss.py --use-ollama  # Use Ollama
    python experiment/06b_generate_tuned_oss.py --use-mlx  # Use MLX (Apple Silicon)
    python experiment/06b_generate_tuned_oss.py --use-transformers  # Use Transformers
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Model paths
FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss-teacher"
MERGED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss-merged"
MLX_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss-mlx"

# Generation config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DEFAULT_BATCH_SIZE = 4  # Adjust based on MacBook memory

# Ollama config
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma-oss-tuned"  # Custom model name in Ollama


def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def fetch_pending_records(supabase, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch records where tuned_oss20b is NULL"""
    print("Fetching records needing tuned_oss20b generation...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        query = supabase.table("modelComp")\
            .select("id, input, context, label")\
            .is_("tuned_oss20b", "null")\
            .not_.is_("label", "null")\
            .order("id")\
            .range(offset, offset + page_size - 1)
        
        response = query.execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched page {offset // page_size + 1}: {len(response.data)} records")
        
        if len(response.data) < page_size:
            break
        
        if limit and len(all_records) >= limit:
            break
        
        offset += page_size
    
    if limit:
        all_records = all_records[:limit]
    
    print(f"✓ Total pending: {len(all_records)} records\n")
    return all_records


def update_tuned_oss(supabase, record_id: int, output: str) -> bool:
    """Update tuned_oss20b column"""
    try:
        supabase.table("modelComp")\
            .update({"tuned_oss20b": output})\
            .eq("id", record_id)\
            .execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Update error for {record_id}: {e}")
        return False


def build_prompt(input_text: str, context: Optional[str] = None) -> str:
    """Build prompt in the same format used during training"""
    if context:
        return f"Context: {context}\n\nQuestion: {input_text}\n\nAnswer:"
    return f"Question: {input_text}\n\nAnswer:"


def prepare_batch(records: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """Prepare a batch of prompts from records"""
    batch = []
    for record in records:
        record_id = record["id"]
        input_text = record["input"]
        context = record.get("context")
        
        # Handle context array
        if isinstance(context, list):
            context = " ".join(str(c) for c in context if c)
        
        prompt = build_prompt(input_text, context)
        batch.append((record_id, prompt))
    
    return batch


# ============================================================================
# TRANSFORMERS GENERATION (Primary method for MacBook)
# ============================================================================

def load_transformers_model(model_path: str):
    """Load model using Transformers with MPS (Apple Silicon)"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading Transformers model from: {model_path}")
        
        # Detect device
        if torch.backends.mps.is_available():
            device_map = {"": "mps"}
            dtype = torch.float16
            print("  Using Apple MPS acceleration")
        else:
            device_map = "auto"
            dtype = torch.float32
            print("  Using CPU")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        
        print("✓ Transformers model loaded\n")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None


def generate_with_transformers_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> List[Optional[str]]:
    """Generate responses using Transformers (BATCH)"""
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        responses = []
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            responses.append(response)
        
        return responses
        
    except Exception as e:
        print(f"\n  ✗ Transformers batch error: {e}")
        return [None] * len(prompts)


def generate_with_transformers_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Optional[str]:
    """Generate response using Transformers (single)"""
    results = generate_with_transformers_batch(model, tokenizer, [prompt], max_new_tokens)
    return results[0] if results else None


# ============================================================================
# MLX GENERATION (Apple Silicon native)
# ============================================================================

def load_mlx_model(model_path: str):
    """Load model using MLX (Apple Silicon)"""
    try:
        from mlx_lm import load, generate
        
        print(f"\nLoading MLX model from: {model_path}")
        model, tokenizer = load(model_path)
        print("✓ MLX model loaded\n")
        return model, tokenizer, generate
    except ImportError:
        print("✗ MLX not installed. Install with: pip install mlx-lm")
        return None, None, None
    except Exception as e:
        print(f"✗ Error loading MLX model: {e}")
        return None, None, None


def generate_with_mlx_single(
    model,
    tokenizer,
    generate_fn,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Optional[str]:
    """Generate response using MLX (single - MLX uses streaming)"""
    try:
        response = generate_fn(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temp=TEMPERATURE,
            top_p=TOP_P,
            verbose=False,
        )
        
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"\n  ✗ MLX error: {e}")
        return None


def generate_with_mlx_batch(
    model,
    tokenizer,
    generate_fn,
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> List[Optional[str]]:
    """Generate responses using MLX (sequential - MLX doesn't support true batch)"""
    responses = []
    for prompt in prompts:
        response = generate_with_mlx_single(model, tokenizer, generate_fn, prompt, max_new_tokens)
        responses.append(response)
    return responses


# ============================================================================
# OLLAMA GENERATION
# ============================================================================

def check_ollama_model(model_name: str = OLLAMA_MODEL) -> bool:
    """Check if custom model exists in Ollama"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = response.json().get("models", [])
        return any(model_name in m.get("name", "") for m in models)
    except:
        return False


def generate_with_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> Optional[str]:
    """Generate response using Ollama (single - Ollama doesn't support batch)"""
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_NEW_TOKENS,
                    "top_p": TOP_P,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"\n  ✗ Ollama error: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tuned_oss20b outputs")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama (no batch)")
    parser.add_argument("--use-mlx", action="store_true", help="Use MLX for Apple Silicon")
    parser.add_argument("--use-transformers", action="store_true", help="Use Transformers")
    parser.add_argument("--model-path", type=str, help="Custom model path")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STEP 06b: GENERATE TUNED-OSS20B OUTPUTS")
    print("=" * 70)
    print("⚠️  Running on MacBook Pro (Apple Silicon)")
    print(f"📦 Batch Size: {args.batch_size}")
    
    # Connect to Supabase
    try:
        supabase = get_supabase_client()
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return
    
    # Fetch pending records
    records = fetch_pending_records(supabase, limit=args.limit)
    
    if not records:
        print("✓ All records already have tuned_oss20b outputs!")
        return
    
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.use_mlx:
        model_path = MLX_MODEL_PATH
    else:
        model_path = FINETUNED_MODEL_PATH
    
    # Initialize model and set generation function
    batch_generate_fn = None
    single_generate_fn = None
    use_batch = True
    
    if args.use_ollama:
        print(f"\nUsing Ollama model: {OLLAMA_MODEL}")
        print("⚠️  Ollama doesn't support batch inference - using single mode")
        if not check_ollama_model():
            print(f"✗ Model '{OLLAMA_MODEL}' not found in Ollama!")
            print("\nTo create, run:")
            print(f"  ollama create {OLLAMA_MODEL} -f Modelfile")
            return
        single_generate_fn = lambda prompt: generate_with_ollama(prompt)
        use_batch = False
        
    elif args.use_mlx:
        model, tokenizer, generate_fn = load_mlx_model(str(model_path))
        if model is None:
            return
        print("⚠️  MLX uses sequential generation (no true batch)")
        batch_generate_fn = lambda prompts: generate_with_mlx_batch(model, tokenizer, generate_fn, prompts)
        single_generate_fn = lambda prompt: generate_with_mlx_single(model, tokenizer, generate_fn, prompt)
        
    elif args.use_transformers:
        model, tokenizer = load_transformers_model(str(model_path))
        if model is None:
            return
        batch_generate_fn = lambda prompts: generate_with_transformers_batch(model, tokenizer, prompts)
        single_generate_fn = lambda prompt: generate_with_transformers_single(model, tokenizer, prompt)
        
    else:
        # Default: Transformers (safest for MacBook)
        print("\nNo backend specified, using Transformers (default for MacBook)")
        model, tokenizer = load_transformers_model(str(model_path))
        if model is None:
            print("Try --use-ollama or --use-mlx instead")
            return
        batch_generate_fn = lambda prompts: generate_with_transformers_batch(model, tokenizer, prompts)
        single_generate_fn = lambda prompt: generate_with_transformers_single(model, tokenizer, prompt)
    
    # Generate outputs
    print(f"\nGenerating {len(records)} tuned_oss20b outputs...")
    if use_batch:
        print(f"🚀 Using BATCH inference (batch_size={args.batch_size})")
    else:
        print("🐢 Using single inference (slower)")
    print("-" * 70)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    if use_batch and batch_generate_fn:
        # BATCH PROCESSING
        for i in tqdm(range(0, len(records), args.batch_size), desc="Batch Generation"):
            batch_records = records[i:i + args.batch_size]
            batch_data = prepare_batch(batch_records)
            
            prompts = [p for _, p in batch_data]
            record_ids = [rid for rid, _ in batch_data]
            
            # Generate batch
            outputs = batch_generate_fn(prompts)
            
            # Update database
            for record_id, output in zip(record_ids, outputs):
                if output:
                    if update_tuned_oss(supabase, record_id, output):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
            
            # Progress update
            total_done = success_count + fail_count
            if total_done % 50 == 0:
                elapsed = time.time() - start_time
                rate = total_done / elapsed * 60
                tqdm.write(f"  ⚡ Rate: {rate:.1f} samples/min | Success: {success_count}")
    else:
        # SINGLE PROCESSING (for Ollama)
        for record in tqdm(records, desc="Single Generation"):
            record_id = record["id"]
            input_text = record["input"]
            context = record.get("context")
            
            if isinstance(context, list):
                context = " ".join(str(c) for c in context if c)
            
            prompt = build_prompt(input_text, context)
            output = single_generate_fn(prompt)
            
            if output:
                if update_tuned_oss(supabase, record_id, output):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
            
            if (success_count + fail_count) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (success_count + fail_count) / elapsed * 60
                tqdm.write(f"  ⚡ Rate: {rate:.1f} samples/min | Success: {success_count}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed:  {fail_count}")
    print(f"⏱  Time:   {elapsed/60:.1f} minutes")
    print(f"⚡ Rate:   {success_count/elapsed*60:.1f} samples/min")
    print("=" * 70)
    print("\n→ Next: Run 07_compare_teachers.py to select best teacher")


if __name__ == "__main__":
    main()
