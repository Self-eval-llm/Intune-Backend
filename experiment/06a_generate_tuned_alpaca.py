"""
Step 06a: Generate Tuned-Alpaca Outputs for Teacher Comparison
==============================================================
Generates outputs from the Gemma model fine-tuned on Alpaca data.
Updates `tuned_alpaca` column in modelComp table.

⚠️ Runs on Windows Laptop (with GPU)

Prerequisites:
- Fine-tuned model at: models/gemma-alpaca-teacher/
- Or merged model at: models/gemma-finetuned-merged/

Usage:
    python experiment/06a_generate_tuned_alpaca.py
    python experiment/06a_generate_tuned_alpaca.py --limit 100  # Test with subset
    python experiment/06a_generate_tuned_alpaca.py --use-merged  # Use merged model
    python experiment/06a_generate_tuned_alpaca.py --use-ollama  # Use Ollama instead
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Windows compatibility - MUST be set before importing unsloth/torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Model paths
FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-alpaca-teacher"
MERGED_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-finetuned-merged"
CHECKPOINT_PATH = FINETUNED_MODEL_PATH / "checkpoint-1233"  # Best checkpoint

# Generation config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
BATCH_SIZE = 1  # Process one at a time for stability

# Ollama config (if using Ollama)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma-alpaca-tuned"  # Custom model name in Ollama


def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def fetch_pending_records(supabase, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch records where tuned_alpaca is NULL"""
    print("Fetching records needing tuned_alpaca generation...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        query = supabase.table("modelComp")\
            .select("id, input, context, label")\
            .is_("tuned_alpaca", "null")\
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


def update_tuned_alpaca(supabase, record_id: int, output: str) -> bool:
    """Update tuned_alpaca column"""
    try:
        supabase.table("modelComp")\
            .update({"tuned_alpaca": output})\
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
# UNSLOTH GENERATION (Primary method for Windows with GPU)
# ============================================================================

def load_unsloth_model(model_path: str):
    """Load fine-tuned model using Unsloth"""
    print(f"\nLoading model from: {model_path}")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    print("✓ Model loaded and ready for inference\n")
    
    return model, tokenizer


def generate_with_unsloth(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate response using Unsloth model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Answer:")
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"\n  ✗ Generation error: {e}")
        return None


# ============================================================================
# OLLAMA GENERATION (Alternative method)
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tuned_alpaca outputs")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument("--use-merged", action="store_true", help="Use merged model")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama instead of Unsloth")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to use")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STEP 06a: GENERATE TUNED-ALPACA OUTPUTS")
    print("=" * 70)
    print("⚠️  Running on Windows Laptop (GPU)")
    
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
        print("✓ All records already have tuned_alpaca outputs!")
        return
    
    # Determine model path
    if args.use_merged:
        model_path = MERGED_MODEL_PATH
    elif args.checkpoint:
        model_path = FINETUNED_MODEL_PATH / args.checkpoint
    else:
        # Use best checkpoint if exists, otherwise base model dir
        if CHECKPOINT_PATH.exists():
            model_path = CHECKPOINT_PATH
        else:
            model_path = FINETUNED_MODEL_PATH
    
    # Initialize model
    if args.use_ollama:
        print(f"\nUsing Ollama model: {OLLAMA_MODEL}")
        if not check_ollama_model():
            print(f"✗ Model '{OLLAMA_MODEL}' not found in Ollama!")
            print("\nTo create, run:")
            print(f"  ollama create {OLLAMA_MODEL} -f Modelfile")
            return
        generate_fn = lambda prompt: generate_with_ollama(prompt)
    else:
        model, tokenizer = load_unsloth_model(str(model_path))
        generate_fn = lambda prompt: generate_with_unsloth(model, tokenizer, prompt)
    
    # Generate outputs
    print(f"Generating {len(records)} tuned_alpaca outputs...")
    print("-" * 70)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for record in tqdm(records, desc="Tuned-Alpaca Generation"):
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
            if update_tuned_alpaca(supabase, record_id, output):
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
    print("\n→ Next: Run 06b_generate_tuned_oss.py on MacBook Pro")
    print("→ Then: Run 07_compare_teachers.py to select best teacher")


if __name__ == "__main__":
    main()
