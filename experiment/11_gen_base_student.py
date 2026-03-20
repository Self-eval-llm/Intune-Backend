"""
Step 1: Generate base student outputs for 50K records (checkpoint by checkpoint)
Uses base Gemma 3:1B model (no finetuning yet) with 4-bit quantization
Saves to student_output column in Supabase - updates after each record

Usage:
  python experiment/11_gen_base_student.py --checkpoint 2    # Generate for checkpoint 2 only
  python experiment/11_gen_base_student.py                   # Generate for lowest incomplete checkpoint
"""

import os
import sys
import time
import argparse

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
RECORDS_PER_CHECKPOINT = 5000

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
    parser = argparse.ArgumentParser(description='Generate base student outputs')
    parser.add_argument('--checkpoint', type=int, default=None, 
                       help='Checkpoint number (1-10). If not specified, auto-detects lowest incomplete.')
    args = parser.parse_args()
    
    # Connect to Supabase
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    
    # Determine which checkpoint to process
    if args.checkpoint:
        target_ckpt = args.checkpoint
        if target_ckpt < 1 or target_ckpt > 10:
            print("Error: Checkpoint must be 1-10")
            sys.exit(1)
    else:
        # Auto-detect: find lowest checkpoint with missing student_output
        print("Auto-detecting checkpoint...")
        for ckpt in range(1, 11):
            result = supabase.table('modelcomp_50k')\
                .select('id', count='exact')\
                .eq('checkpoint', ckpt)\
                .is_('student_output', 'null')\
                .execute()
            if result.count > 0:
                target_ckpt = ckpt
                print(f"  Checkpoint {ckpt}: {result.count} records pending")
                break
        else:
            print("✅ All checkpoints complete!")
            return
    
    # Check pending for this checkpoint
    result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', target_ckpt)\
        .is_('student_output', 'null')\
        .execute()
    pending_count = result.count
    
    # Check already done
    done_result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', target_ckpt)\
        .not_.is_('student_output', 'null')\
        .execute()
    done_count = done_result.count
    
    print(f"\n{'='*60}")
    print(f"📋 CHECKPOINT {target_ckpt} STATUS")
    print(f"{'='*60}")
    print(f"   Already done:    {done_count:,} / {RECORDS_PER_CHECKPOINT:,}")
    print(f"   Pending:         {pending_count:,}")
    print(f"   Progress:        {done_count/RECORDS_PER_CHECKPOINT*100:.1f}%")
    print(f"{'='*60}")
    
    if pending_count == 0:
        print(f"✅ Checkpoint {target_ckpt} already complete!")
        print(f"   Next: python experiment/11_gen_base_student.py --checkpoint {target_ckpt + 1}")
        return
    
    # Load model
    model, tokenizer = load_model()
    
    # Process in batches (only for this checkpoint)
    total_processed = 0
    
    while True:
        # Fetch batch without student_output FOR THIS CHECKPOINT
        batch = supabase.table('modelcomp_50k')\
            .select('id, input, context')\
            .eq('checkpoint', target_ckpt)\
            .is_('student_output', 'null')\
            .order('id')\
            .limit(BATCH_SIZE)\
            .execute()
        
        if not batch.data:
            break
        
        print(f"\nProcessing batch of {len(batch.data)} records (Checkpoint {target_ckpt})...")
        
        for record in tqdm(batch.data, desc=f"Ckpt {target_ckpt}"):
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
        
        print(f"Checkpoint {target_ckpt} progress: {done_count + total_processed}/{RECORDS_PER_CHECKPOINT}")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    print(f"\n✅ Checkpoint {target_ckpt} done! Generated {total_processed} student outputs")
    print(f"   Total for checkpoint: {done_count + total_processed}/{RECORDS_PER_CHECKPOINT}")
    
    if done_count + total_processed >= RECORDS_PER_CHECKPOINT:
        print(f"\n   🎯 Checkpoint {target_ckpt} is READY for training!")
        print(f"   Run: python experiment/12_train_incremental.py --checkpoint {target_ckpt} --init")
    
    if target_ckpt < 10:
        print(f"   Next: python experiment/11_gen_base_student.py --checkpoint {target_ckpt + 1}")

if __name__ == "__main__":
    main()
