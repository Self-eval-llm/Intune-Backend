"""
Step 06b: Generate Tuned-OSS20B Outputs for Teacher Comparison
==============================================================
Generates outputs from the Gemma model fine-tuned on OSS-20B data.
Updates `tuned_oss20b` column in modelComp table.

⚠️ Runs on MacBook Pro (24GB Apple Silicon)

Prerequisites:
- Fine-tuned model at: models/gemma-oss20b-teacher/
- Or via Ollama: gemma-oss-tuned

Usage:
    python experiment/06b_generate_tuned_oss.py
    python experiment/06b_generate_tuned_oss.py --limit 100  # Test with subset
    python experiment/06b_generate_tuned_oss.py --use-ollama --model gemma-oss-tuned
    python experiment/06b_generate_tuned_oss.py --use-mlx  # Use MLX for Apple Silicon
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Model paths
FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss20b-teacher"
MERGED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss20b-merged"

# Generation config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Ollama config
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma-oss-tuned"  # Custom model name


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


# ============================================================================
# OLLAMA GENERATION (Primary method for MacBook)
# ============================================================================

def check_ollama_model(model_name: str = OLLAMA_MODEL) -> bool:
    """Check if custom model exists in Ollama"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        print(f"  Available models: {model_names[:5]}...")
        return any(model_name in m for m in model_names)
    except Exception as e:
        print(f"  Connection error: {e}")
        return False


def generate_with_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> Optional[str]:
    """Generate response using Ollama"""
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
# MLX GENERATION (Alternative for Apple Silicon)
# ============================================================================

def load_mlx_model(model_path: str):
    """Load model using MLX for Apple Silicon"""
    try:
        from mlx_lm import load, generate
        print(f"\nLoading MLX model from: {model_path}")
        model, tokenizer = load(model_path)
        print("✓ MLX model loaded\n")
        return model, tokenizer
    except ImportError:
        print("✗ MLX not installed. Install with: pip install mlx-lm")
        return None, None


def generate_with_mlx(model, tokenizer, prompt: str) -> Optional[str]:
    """Generate response using MLX"""
    try:
        from mlx_lm import generate
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=MAX_NEW_TOKENS,
            temp=TEMPERATURE,
            top_p=TOP_P,
        )
        
        # Extract only the generated part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"\n  ✗ MLX error: {e}")
        return None


# ============================================================================
# TRANSFORMERS GENERATION (Fallback)
# ============================================================================

def load_transformers_model(model_path: str):
    """Load model using Transformers (slower but compatible)"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading Transformers model from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print("✓ Transformers model loaded\n")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None


def generate_with_transformers(model, tokenizer, prompt: str) -> Optional[str]:
    """Generate response using Transformers"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"\n  ✗ Transformers error: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tuned_oss20b outputs")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama (recommended)")
    parser.add_argument("--use-mlx", action="store_true", help="Use MLX for Apple Silicon")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Model name")
    parser.add_argument("--model-path", type=str, help="Path to model (for MLX/Transformers)")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STEP 06b: GENERATE TUNED-OSS20B OUTPUTS")
    print("=" * 70)
    print("⚠️  Running on MacBook Pro (Apple Silicon)")
    
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
    
    # Determine generation method
    model = None
    tokenizer = None
    
    if args.use_ollama:
        print(f"\nUsing Ollama model: {args.model}")
        if not check_ollama_model(args.model):
            print(f"\n✗ Model '{args.model}' not found in Ollama!")
            print("\nTo create a custom model, create a Modelfile and run:")
            print(f"  ollama create {args.model} -f Modelfile")
            print("\nOr use an existing model:")
            print("  python 06b_generate_tuned_oss.py --use-ollama --model gemma2:2b")
            return
        generate_fn = lambda prompt: generate_with_ollama(prompt, args.model)
        
    elif args.use_mlx:
        model_path = args.model_path or str(MERGED_MODEL_PATH)
        model, tokenizer = load_mlx_model(model_path)
        if model is None:
            return
        generate_fn = lambda prompt: generate_with_mlx(model, tokenizer, prompt)
        
    else:
        # Default to Ollama as it's easiest on Mac
        print("\nDefaulting to Ollama (recommended for MacBook)")
        print("Use --use-mlx for MLX or --use-ollama for explicit Ollama")
        
        if not check_ollama_model(args.model):
            print(f"\n✗ Model '{args.model}' not found!")
            print("\nOptions:")
            print("  1. Create custom Ollama model from fine-tuned weights")
            print("  2. Use existing model: --model gemma2:2b")
            print("  3. Use MLX: --use-mlx --model-path path/to/model")
            return
        generate_fn = lambda prompt: generate_with_ollama(prompt, args.model)
    
    # Generate outputs
    print(f"\nGenerating {len(records)} tuned_oss20b outputs...")
    print("-" * 70)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for record in tqdm(records, desc="Tuned-OSS20B Generation"):
        record_id = record["id"]
        input_text = record["input"]
        context = record.get("context")
        
        # Handle context array
        if isinstance(context, list):
            context = " ".join(str(c) for c in context if c)
        
        # Build prompt
        prompt = build_prompt(input_text, context)
        
        # Generate
        output = generate_fn(prompt)
        
        if output:
            if update_tuned_oss(supabase, record_id, output):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1
        
        # Progress update
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
    print("\n→ Next: Run 07_compare_teachers.py to compare both teachers")
    print("→ Then: Select the better teacher for 50K Gemma 3 fine-tuning")


if __name__ == "__main__":
    main()
