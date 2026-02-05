"""
Step 1: Generate base student outputs for all 50K records
Uses base Gemma 3:1B model (no finetuning yet) with 4-bit quantization
Saves to student_output column in Supabase - updates after each record
"""

import os
import time

# Disable torch dynamo/compile completely to avoid Triton issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from tqdm import tqdm
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Config - use 4bit for speed
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
BATCH_SIZE = 50  # Records per Supabase fetch
MAX_NEW_TOKENS = 256  # Shorter outputs for speed

def load_model():
    """Load base Gemma model with 4-bit quantization using Unsloth"""
    print(f"Loading {MODEL_NAME} (4-bit)...")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    print(f"Model loaded on {model.device}")
    return model, tokenizer

def generate_output(model, tokenizer, instruction, context=""):
    """Generate output for a single prompt"""
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
    latency = (time.time() - start_time) * 1000  # ms
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response, latency

def main():
    # Connect to Supabase
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    
    # Check how many already have student_output
    result = supabase.table('modelcomp_50k').select('id', count='exact').is_('student_output', 'null').execute()
    pending_count = result.count
    print(f"Records needing student_output: {pending_count}")
    
    if pending_count == 0:
        print("All records already have student_output!")
        return
    
    # Load model
    model, tokenizer = load_model()
    
    # Process in batches
    total_processed = 0
    
    while True:
        # Fetch batch without student_output
        batch = supabase.table('modelcomp_50k')\
            .select('id, input, context')\
            .is_('student_output', 'null')\
            .order('id')\
            .limit(BATCH_SIZE)\
            .execute()
        
        if not batch.data:
            break
        
        print(f"\nProcessing batch of {len(batch.data)} records...")
        
        for record in tqdm(batch.data, desc="Generating"):
            try:
                output, latency = generate_output(
                    model, tokenizer,
                    record['input'],
                    record.get('context', '')
                )
                
                # Update Supabase immediately after each record
                supabase.table('modelcomp_50k').update({
                    'student_output': output[:5000],
                    'generation_latency': round(latency, 3)
                }).eq('id', record['id']).execute()
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error on record {record['id']}: {e}")
                continue
        
        print(f"Total processed: {total_processed}/{pending_count}")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    print(f"\n✅ Done! Generated student outputs for {total_processed} records")

if __name__ == "__main__":
    main()
