"""
Step 08: Generate Context for Records Missing Context
======================================================
Uses OSS-20B (via Ollama/API) or HuggingFace Alpaca to generate 
contextual information for records that have NULL context.

This improves evaluation accuracy for context-dependent metrics:
- hallucination_score
- context_grounding_score
- faithfulness (context component)

Usage:
    python experiment/08_generate_context.py --limit 100  # Test
    python experiment/08_generate_context.py --use-ollama  # Use local Ollama
    python experiment/08_generate_context.py --use-hf  # Use HuggingFace API
    python experiment/08_generate_context.py --batch-size 8  # Batch processing
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

# Ollama config
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"  # or "mistral", "gemma", etc.

# HuggingFace config
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # or alpaca model

# Generation settings
MAX_CONTEXT_TOKENS = 256
TEMPERATURE = 0.3  # Low temp for factual context


def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def fetch_null_context_records(supabase, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch records where context is NULL"""
    print("Fetching records with NULL context...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        # Fetch records where context is NULL or empty
        query = supabase.table("modelComp")\
            .select("id, input, actual_output, label")\
            .is_("context", "null")\
            .order("id")\
            .range(offset, offset + page_size - 1)
        
        response = query.execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched {len(all_records)} records...")
        
        if len(response.data) < page_size:
            break
        
        if limit and len(all_records) >= limit:
            break
        
        offset += page_size
    
    if limit:
        all_records = all_records[:limit]
    
    print(f"✓ Total NULL context: {len(all_records)} records\n")
    return all_records


def update_context(supabase, record_id: str, context: str) -> bool:
    """Update context column"""
    try:
        supabase.table("modelComp")\
            .update({"context": context})\
            .eq("id", record_id)\
            .execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Update error for {record_id}: {e}")
        return False


def build_context_prompt(instruction: str, answer: str, label: str) -> str:
    """
    Build a prompt to generate relevant context.
    The context should provide background information that would help
    answer the question correctly.
    """
    prompt = f"""Given this question and its correct answer, generate a brief factual context 
(2-3 sentences) that provides background information relevant to answering the question.

Task Category: {label}
Question: {instruction}
Correct Answer: {answer}

Generate ONLY the context (no explanations, no labels):
Context:"""
    return prompt


# ============================================================================
# OLLAMA GENERATION
# ============================================================================

def check_ollama_available() -> bool:
    """Check if Ollama is running"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def generate_context_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Generate context using Ollama"""
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_CONTEXT_TOKENS,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        
        # Clean up - remove "Context:" prefix if present
        if result.lower().startswith("context:"):
            result = result[8:].strip()
        
        return result if result else None
        
    except Exception as e:
        print(f"\n  ✗ Ollama error: {e}")
        return None


# ============================================================================
# HUGGINGFACE GENERATION
# ============================================================================

def load_hf_model(model_name: str = HF_MODEL):
    """Load HuggingFace model"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading HuggingFace model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print("✓ Model loaded\n")
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None


def generate_context_hf_batch(
    model,
    tokenizer,
    prompts: List[str],
) -> List[Optional[str]]:
    """Generate context using HuggingFace (batch)"""
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_CONTEXT_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        results = []
        for i, output in enumerate(outputs):
            # Decode only new tokens
            input_len = inputs['input_ids'][i].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            
            # Clean up
            if response.lower().startswith("context:"):
                response = response[8:].strip()
            
            results.append(response if response else None)
        
        return results
        
    except Exception as e:
        print(f"\n  ✗ HF batch error: {e}")
        return [None] * len(prompts)


# ============================================================================
# SIMPLE CONTEXT GENERATION (No external model needed)
# ============================================================================

def generate_context_simple(instruction: str, answer: str, label: str) -> str:
    """
    Generate simple synthetic context based on the question and answer.
    This doesn't require any external model - just reformulates the answer
    as background information.
    
    Good for bootstrapping when you don't have access to models.
    """
    # Task-specific context templates
    templates = {
        "technical_code": f"This question relates to programming and software development. The expected solution involves: {answer[:200]}...",
        "math_logic": f"This is a mathematical or logical reasoning problem. The correct approach leads to: {answer[:200]}...",
        "classification_analysis": f"This task requires classification or analysis. The key points to identify are related to: {answer[:200]}...",
        "language_editing": f"This is a language and text editing task. The correct form should be: {answer[:200]}...",
        "creative_generative": f"This creative task requires generating content. A good response would include elements like: {answer[:200]}...",
        "general_qa": f"Background information: {answer[:300]}...",
    }
    
    # Clean label (remove quotes if present)
    clean_label = label.strip('"\'') if label else "general_qa"
    
    return templates.get(clean_label, templates["general_qa"])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate context for NULL records")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace")
    parser.add_argument("--use-simple", action="store_true", help="Use simple template-based context")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--hf-model", type=str, default=HF_MODEL, help="HuggingFace model name")
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB updates")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STEP 08: GENERATE CONTEXT FOR NULL RECORDS")
    print("=" * 70)
    
    # Connect to Supabase
    try:
        supabase = get_supabase_client()
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return
    
    # Fetch records
    records = fetch_null_context_records(supabase, limit=args.limit)
    
    if not records:
        print("✓ No records with NULL context!")
        return
    
    # Determine generation method
    generate_fn = None
    use_batch = False
    model, tokenizer = None, None
    
    if args.use_simple:
        print("\nUsing SIMPLE template-based context generation")
        print("⚠️  No external model needed - fast but less accurate")
        generate_fn = lambda r: generate_context_simple(
            r["input"], r.get("actual_output", ""), r.get("label", "general_qa")
        )
        
    elif args.use_ollama:
        print(f"\nUsing Ollama model: {args.ollama_model}")
        if not check_ollama_available():
            print("✗ Ollama not running! Start with: ollama serve")
            return
        generate_fn = lambda prompt: generate_context_ollama(prompt, args.ollama_model)
        
    elif args.use_hf:
        model, tokenizer = load_hf_model(args.hf_model)
        if model is None:
            return
        use_batch = True
        
    else:
        # Default: simple (no model required)
        print("\nNo method specified, using SIMPLE template-based context")
        print("Use --use-ollama or --use-hf for model-based generation")
        generate_fn = lambda r: generate_context_simple(
            r["input"], r.get("actual_output", ""), r.get("label", "general_qa")
        )
    
    # Generate context
    print(f"\nGenerating context for {len(records)} records...")
    if args.dry_run:
        print("⚠️  DRY RUN - No database updates")
    print("-" * 70)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    if use_batch and model is not None:
        # Batch processing for HuggingFace
        for i in tqdm(range(0, len(records), args.batch_size), desc="Batch Generation"):
            batch = records[i:i + args.batch_size]
            
            # Build prompts
            prompts = [
                build_context_prompt(
                    r["input"],
                    r.get("actual_output", ""),
                    r.get("label", "general_qa")
                )
                for r in batch
            ]
            
            # Generate
            contexts = generate_context_hf_batch(model, tokenizer, prompts)
            
            # Update database
            for record, context in zip(batch, contexts):
                if context:
                    if args.dry_run or update_context(supabase, record["id"], context):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
    else:
        # Single processing
        for record in tqdm(records, desc="Generating Context"):
            if args.use_simple or (not args.use_ollama and not args.use_hf):
                # Simple template
                context = generate_fn(record)
            else:
                # Ollama
                prompt = build_context_prompt(
                    record["input"],
                    record.get("actual_output", ""),
                    record.get("label", "general_qa")
                )
                context = generate_fn(prompt)
            
            if context:
                if args.dry_run or update_context(supabase, record["id"], context):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("CONTEXT GENERATION COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed:  {fail_count}")
    print(f"⏱  Time:   {elapsed/60:.1f} minutes")
    print(f"⚡ Rate:   {success_count/elapsed*60:.1f} records/min")
    print("=" * 70)
    print("\n→ Next: Run 07_compare_teachers.py to evaluate with context")


if __name__ == "__main__":
    main()
