"""
Incremental Learning Pipeline with Status-Based Workflow
=========================================================
Status Flow for each checkpoint (5000 records):
  score        -> Score base student_output vs teacher (all metrics)
  finetune     -> Finetune model on this checkpoint's data  
  output_tuned -> Generate student_output_tuned with finetuned model
  score_tuned  -> Score tuned output vs teacher (all metrics with _tuned suffix)
  completed    -> Calculate improvement for each metric

IMPORTANT: Each step validates that 5000 records exist with the required status
           before proceeding. Records start with status='score'.

Columns used:
  - input, context: Input data
  - sevenb: Teacher output (expected output)
  - student_output: Base model output
  - student_output_tuned: Finetuned model output
  - latency, latency_tuned: Inference times
  - checkpoint: Checkpoint number (1-10)
  - status: Workflow status
  
  Before Finetuning (base student):
  - score, structured_correctness, task_success, instruction_following
  - coverage, faithfulness, hallucination, context_grounding, conciseness
  - rouge1, rougel, bleu
  
  After Finetuning (tuned student):
  - score_tuned, structured_correctness_tuned, task_success_tuned, instruction_following_tuned
  - coverage_tuned, faithfulness_tuned, hallucination_tuned, context_grounding_tuned, conciseness_tuned
  - rouge1_tuned, rougel_tuned, bleu_tuned
  
  Improvement:
  - improvement (score_tuned - score)

Usage:
    # Check status of a checkpoint
    python experiment/12_train_incremental.py --checkpoint 1 --step status
    
    # Run individual steps
    python experiment/12_train_incremental.py --checkpoint 1 --step score
    python experiment/12_train_incremental.py --checkpoint 1 --step finetune
    python experiment/12_train_incremental.py --checkpoint 1 --step output_tuned
    python experiment/12_train_incremental.py --checkpoint 1 --step score_tuned
    python experiment/12_train_incremental.py --checkpoint 1 --step completed
    
    # Run all steps for a checkpoint automatically
    python experiment/12_train_incremental.py --checkpoint 1 --run-all
    
    # Initialize checkpoint (set status to 'score')
    python experiment/12_train_incremental.py --checkpoint 1 --init
"""

import os
import sys
import json
import time
import argparse

# Fix Unicode/Emoji encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load .env before any other imports
from dotenv import load_dotenv
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

# Import other modules
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
from supabase import create_client

# Training imports
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Disable torch.compile for Windows compatibility
torch._dynamo.config.disable = True          # ← ADD THIS
torch._dynamo.config.suppress_errors = True
torch._inductor.config.cpp_wrapper = False

# Evaluation imports - use comprehensive metrics from 06_eval_metrics
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
from importlib import import_module
eval_metrics = import_module('06_eval_metrics')
evaluate_single_output = eval_metrics.evaluate_single_output
compare_teacher_student = eval_metrics.compare_teacher_student

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Config
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048
RECORDS_PER_CHECKPOINT = 5000
MAX_NEW_TOKENS = 512

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Status flow (removed 'eo' - start from 'score')
STATUS_FLOW = ['score', 'finetune', 'output_tuned', 'score_tuned', 'completed']
# Min threshold (lower than 5000 to handle uneven checkpoint sizes like 4978-5004)
# Lowered to 4850 to allow 98%+ completion (handles last checkpoint records)
MIN_RECORDS_PER_CHECKPOINT = 4800

def get_supabase():
    """Get Supabase client"""
    return create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def get_checkpoint_size(supabase, checkpoint):
    """Get actual number of records in a checkpoint (may not be exactly 5000)"""
    result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', checkpoint)\
        .execute()
    return result.count

def fetch_records_by_status(supabase, checkpoint, status):
    """Fetch records for a checkpoint with specific status (with pagination for >1000 records)"""
    print(f"\nFetching checkpoint {checkpoint} records with status='{status}'...")
    
    all_records = []
    offset = 0
    batch_size = 1000
    
    while True:
        result = supabase.table('modelcomp_50k')\
            .select('*')\
            .eq('checkpoint', checkpoint)\
            .eq('status', status)\
            .range(offset, offset + batch_size - 1)\
            .execute()
        
        if not result.data:
            break
        
        all_records.extend(result.data)
        offset += batch_size
        
        if len(result.data) < batch_size:
            break
    
    print(f"Fetched {len(all_records)} records")
    return all_records

def validate_records_count(records, step_name, min_required=MIN_RECORDS_PER_CHECKPOINT):
    """Validate that we have minimum required records for a step.
    Returns True if validation passed, False otherwise.
    Always prints status of ready/remaining records.
    """
    count = len(records)
    remaining = max(0, min_required - count)
    pct_ready = (count / min_required) * 100 if min_required > 0 else 0
    
    print(f"\n{'─'*50}")
    print(f"📊 DATA READINESS CHECK: {step_name}")
    print(f"{'─'*50}")
    print(f"   Records ready:     {count:,}")
    print(f"   Records required:  {min_required:,}")
    print(f"   Records remaining: {remaining:,}")
    print(f"   Progress:          {pct_ready:.1f}%")
    
    # Visual progress bar
    bar_length = 40
    filled = int(bar_length * count / min_required) if min_required > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"   [{bar}]")
    print(f"{'─'*50}")
    
    if count < min_required:
        print(f"\n❌ VALIDATION FAILED: {step_name}")
        print(f"   Need {remaining:,} more records to proceed.")
        print(f"   Current: {count:,} / {min_required:,} ({pct_ready:.1f}% ready)")
        print(f"\n   💡 Collect more data and run --init again, or wait for more records.")
        return False
    
    print(f"✅ Validation passed: {count:,} records ready for {step_name}")
    return True

def fetch_checkpoint_records(supabase, checkpoint):
    """Fetch all records for a checkpoint"""
    print(f"\nFetching all records for checkpoint {checkpoint}...")
    
    result = supabase.table('modelcomp_50k')\
        .select('*')\
        .eq('checkpoint', checkpoint)\
        .execute()
    
    print(f"Fetched {len(result.data)} records")
    return result.data

def update_status(supabase, record_ids, new_status):
    """Update status for multiple records with retry logic"""
    print(f"\nUpdating {len(record_ids)} records to status='{new_status}'...")
    
    import time
    for i, rid in enumerate(tqdm(record_ids, desc="Updating status")):
        retries = 3
        while retries > 0:
            try:
                supabase.table('modelcomp_50k').update({
                    'status': new_status
                }).eq('id', rid).execute()
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"\n⚠️ Failed to update record {rid}: {e}")
                else:
                    time.sleep(1)  # Wait before retry
        
        # Slow down to avoid rate limiting (every 100 records)
        if (i + 1) % 100 == 0:
            time.sleep(0.5)

def calculate_metrics(prediction, reference, instruction="", context="", task_label="general_qa"):
    """
    Calculate comprehensive evaluation metrics using 06_eval_metrics module.
    
    Returns dict with all metrics:
    - structured_correctness
    - task_success
    - instruction_following
    - coverage
    - faithfulness
    - hallucination
    - context_grounding
    - conciseness
    - overall_score (weighted combination)
    """
    if not prediction or not reference:
        return {
            'overall': 0.0,
            'structured_correctness': 0.0,
            'task_success': 0.0,
            'instruction_following': 0.0,
            'coverage': 0.0,
            'faithfulness': 0.0,
            'hallucination': 0.0,
            'context_grounding': 0.0,
            'conciseness': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougel': 0.0,
            'bleu': 0.0
        }
    
    # Use comprehensive evaluation from 06_eval_metrics
    eval_result = evaluate_single_output(
        instruction=instruction,
        student_output=prediction,
        teacher_output=reference,
        context=context,
        task_label=task_label
    )
    
    # Also calculate basic ROUGE/BLEU for backwards compatibility
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  # ROUGE scorer uses uppercase L
    rouge_scores = scorer.score(reference, prediction)
    
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    except:
        bleu = 0.0
    
    return {
        # Comprehensive metrics
        'overall': eval_result['overall_score'],
        'structured_correctness': eval_result['structured_correctness'],
        'task_success': eval_result['task_success'],
        'instruction_following': eval_result['instruction_following'],
        'coverage': eval_result['coverage'],
        'faithfulness': eval_result['faithfulness'],
        'hallucination': eval_result['hallucination'],
        'context_grounding': eval_result['context_grounding'],
        'conciseness': eval_result['conciseness'],
        # Basic metrics
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougel': rouge_scores['rougeL'].fmeasure,  # API returns uppercase, store as lowercase
        'bleu': bleu,
        # Details
        'details': eval_result.get('details', {})
    }

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

# =============================================================================
# STATUS CHECK - View current checkpoint status
# =============================================================================
def step_status(supabase, checkpoint):
    """Show status summary for a checkpoint with detailed readiness info"""
    print(f"\n{'='*60}")
    print(f"📊 STATUS SUMMARY - Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    # Count records by status
    all_records = fetch_checkpoint_records(supabase, checkpoint)
    
    status_counts = {}
    for status in STATUS_FLOW + [None, '']:
        status_counts[status if status else 'null/empty'] = 0
    
    for r in all_records:
        s = r.get('status')
        if s in status_counts:
            status_counts[s] += 1
        elif not s:
            status_counts['null/empty'] += 1
        else:
            status_counts[s] = status_counts.get(s, 0) + 1
    
    print(f"\n   Total records: {len(all_records)}")
    print(f"   Required per step: {MIN_RECORDS_PER_CHECKPOINT}")
    print(f"\n   Status Distribution:")
    print(f"   {'─'*50}")
    
    # Find the active step (first one with records but not enough)
    next_step = None
    ready_count = 0
    
    for status in ['null/empty'] + STATUS_FLOW:
        count = status_counts.get(status, 0)
        pct = (count / MIN_RECORDS_PER_CHECKPOINT) * 100 if MIN_RECORDS_PER_CHECKPOINT > 0 else 0
        
        # Visual progress bar for each status
        bar_length = 30
        filled = int(bar_length * min(count / MIN_RECORDS_PER_CHECKPOINT, 1)) if MIN_RECORDS_PER_CHECKPOINT > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        ready = '✅' if count >= MIN_RECORDS_PER_CHECKPOINT else '⏳' if count > 0 else '❌'
        print(f"   {ready} {status:15} : {count:5} / {MIN_RECORDS_PER_CHECKPOINT} [{bar}] {pct:.0f}%")
        
        # Track the next actionable step
        if status in STATUS_FLOW and count > 0 and next_step is None:
            next_step = status
            ready_count = count
    
    # Detailed readiness summary
    print(f"\n   {'─'*50}")
    print(f"   📈 READINESS SUMMARY")
    print(f"   {'─'*50}")
    
    if next_step:
        remaining = max(0, MIN_RECORDS_PER_CHECKPOINT - ready_count)
        pct_ready = (ready_count / MIN_RECORDS_PER_CHECKPOINT) * 100
        
        print(f"   Next step:         --step {next_step}")
        print(f"   Records ready:     {ready_count:,}")
        print(f"   Records remaining: {remaining:,}")
        print(f"   Progress:          {pct_ready:.1f}%")
        
        if ready_count >= MIN_RECORDS_PER_CHECKPOINT:
            print(f"\n   ✅ READY TO RUN: python experiment/12_train_incremental.py --checkpoint {checkpoint} --step {next_step}")
        else:
            print(f"\n   ⏳ WAITING: Need {remaining:,} more records for '{next_step}' step")
            print(f"      Collect more data or wait for pipeline to accumulate records.")
    else:
        null_count = status_counts.get('null/empty', 0)
        if null_count > 0:
            remaining = max(0, MIN_RECORDS_PER_CHECKPOINT - null_count)
            print(f"   Uninitialized:     {null_count:,}")
            print(f"   Records remaining: {remaining:,}")
            print(f"\n   ⚠️  Run --init to initialize {null_count:,} records")
            if null_count < MIN_RECORDS_PER_CHECKPOINT:
                print(f"      Then collect {remaining:,} more records before running --step score")
        else:
            completed = status_counts.get('completed', 0)
            if completed >= MIN_RECORDS_PER_CHECKPOINT:
                print(f"   ✅ Checkpoint {checkpoint} COMPLETED!")
                if checkpoint < 10:
                    print(f"      Ready for checkpoint {checkpoint + 1}")
    
    print(f"{'='*60}")
    return status_counts

# =============================================================================
# INIT - Initialize checkpoint records to 'score' status
# =============================================================================
def init_checkpoint(supabase, checkpoint):
    """Initialize checkpoint by setting status to 'score' for records without status"""
    print(f"\n{'='*60}")
    print(f"🔧 INIT - Initialize Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    # Fetch records with null/empty status (paginate to get all, Supabase limits to 1000)
    all_records = []
    offset = 0
    batch_size = 1000
    
    while True:
        result = supabase.table('modelcomp_50k')\
            .select('id, sevenb, student_output')\
            .eq('checkpoint', checkpoint)\
            .or_('status.is.null,status.eq.')\
            .range(offset, offset + batch_size - 1)\
            .execute()
        
        if not result.data:
            break
        
        all_records.extend(result.data)
        offset += batch_size
        
        if len(result.data) < batch_size:
            break
    
    records = all_records
    print(f"Found {len(records)} records without status")
    
    # Validate they have required data
    valid_records = []
    missing_sevenb = 0
    missing_student = 0
    
    for r in records:
        if not r.get('sevenb'):
            missing_sevenb += 1
        elif not r.get('student_output'):
            missing_student += 1
        else:
            valid_records.append(r['id'])
    
    if missing_sevenb > 0:
        print(f"⚠️  {missing_sevenb} records missing teacher output (sevenb)")
    if missing_student > 0:
        print(f"⚠️  {missing_student} records missing base student output")
    
    # Show readiness status
    count = len(valid_records)
    total_in_checkpoint = get_checkpoint_size(supabase, checkpoint)
    # Use actual checkpoint size as target (handles uneven checkpoints like 4978-5004)
    required = max(MIN_RECORDS_PER_CHECKPOINT, total_in_checkpoint - missing_sevenb)
    remaining = max(0, required - count)
    pct_ready = (count / required) * 100 if required > 0 else 0
    
    print(f"\n{'─'*50}")
    print(f"📊 DATA READINESS FOR CHECKPOINT {checkpoint}")
    print(f"{'─'*50}")
    print(f"   Total in checkpoint: {total_in_checkpoint:,}")
    print(f"   Records ready:      {count:,}")
    print(f"   Records remaining:  {remaining:,}")
    print(f"   Progress:           {pct_ready:.1f}%")
    
    # Visual progress bar
    bar_length = 40
    filled = int(bar_length * pct_ready / 100) if required > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"   [{bar}]")
    print(f"{'─'*50}")
    
    if valid_records:
        print(f"\n✅ {len(valid_records)} records ready to initialize")
        update_status(supabase, valid_records, 'score')
        print(f"✅ Status set to 'score' - ready for scoring")
        
        if count < MIN_RECORDS_PER_CHECKPOINT:
            print(f"\n⚠️  WARNING: Only {count:,} records initialized.")
            print(f"   Need {remaining:,} more records before running --step score")
            print(f"   Generate more with: python experiment/11_gen_base_student.py --checkpoint {checkpoint}")
    else:
        print("❌ No valid records to initialize")
    
    return len(valid_records)

# =============================================================================
# STEP 1: score - Score Base Student Output
# =============================================================================
def step_score(supabase, checkpoint):
    """Score base student_output against teacher (sevenb) using full evaluation matrix"""
    print(f"\n{'='*60}")
    print(f"📊 STEP: score - Score Base Student for Checkpoint {checkpoint}")
    print(f"   Using comprehensive evaluation metrics...")
    print(f"{'='*60}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'score')
    
    # Validate minimum records
    if not validate_records_count(records, 'score'):
        return 0
    
    all_metrics = []
    for item in tqdm(records, desc="Scoring with full matrix"):
        student_out = item.get('student_output', '')
        teacher_out = item.get('sevenb', '')
        instruction = item.get('input', '')
        context = item.get('context', '')
        task_label = item.get('task_label', 'general_qa')
        
        if student_out and teacher_out:
            metrics = calculate_metrics(
                prediction=student_out,
                reference=teacher_out,
                instruction=instruction,
                context=context,
                task_label=task_label
            )
            score = round(metrics['overall'], 4)
        else:
            metrics = None
            score = None
        
        # Update all scores in Supabase as individual columns
        update_data = {
            'score': score,
            'status': 'finetune'
        }
        
        # Store ALL metrics as individual columns (before finetuning)
        if metrics:
            update_data['structured_correctness'] = round(metrics['structured_correctness'], 4)
            update_data['task_success'] = round(metrics['task_success'], 4)
            update_data['instruction_following'] = round(metrics['instruction_following'], 4)
            update_data['coverage'] = round(metrics['coverage'], 4)
            update_data['faithfulness'] = round(metrics['faithfulness'], 4)
            update_data['hallucination'] = round(metrics['hallucination'], 4)
            update_data['context_grounding'] = round(metrics['context_grounding'], 4)
            update_data['conciseness'] = round(metrics['conciseness'], 4)
            update_data['rouge1'] = round(metrics['rouge1'], 4)
            update_data['rougel'] = round(metrics['rougel'], 4)
            update_data['bleu'] = round(metrics['bleu'], 4)
        
        supabase.table('modelcomp_50k').update(update_data).eq('id', item['id']).execute()
        
        if metrics:
            all_metrics.append(metrics)
    
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"📊 BASE STUDENT EVALUATION SUMMARY (Checkpoint {checkpoint})")
        print(f"{'='*60}")
        print(f"   Records evaluated: {len(all_metrics)}")
        print(f"   ─────────────────────────────────────")
        print(f"   Overall Score:         {sum(m['overall'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ─────────────────────────────────────")
        print(f"   Structured Correctness: {sum(m['structured_correctness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Task Success:           {sum(m['task_success'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Instruction Following:  {sum(m['instruction_following'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Coverage:               {sum(m['coverage'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Faithfulness:           {sum(m['faithfulness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Hallucination:          {sum(m['hallucination'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Context Grounding:      {sum(m['context_grounding'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Conciseness:            {sum(m['conciseness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ─────────────────────────────────────")
        print(f"   ROUGE-1:                {sum(m['rouge1'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ROUGE-L:                {sum(m['rougel'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   BLEU:                   {sum(m['bleu'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"{'='*60}")
    
    avg_score = sum(m['overall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    print(f"\n✅ Status updated to 'finetune' - ready for finetuning")
    
    return avg_score

# =============================================================================
# STEP 3: finetune - Finetune Model on Checkpoint Data
# =============================================================================
def step_finetune(supabase, checkpoint):
    """Finetune model on checkpoint data"""
    print(f"\n{'='*60}")
    print(f"🔧 STEP: finetune - Train on Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'finetune')
    
    # Validate minimum records
    if not validate_records_count(records, 'finetune'):
        return None
    
    # Format training data
    formatted = []
    for item in records:
        context = item.get('context') or ''
        if context:
            text = f"### Instruction:\n{item['input']}\n\n### Context:\n{context}\n\n### Response:\n{item['sevenb']}"
        else:
            text = f"### Instruction:\n{item['input']}\n\n### Response:\n{item['sevenb']}"
        formatted.append({"text": text})
    
    dataset = Dataset.from_list(formatted)
    print(f"Training on {len(dataset)} records")
    
    # Load model (previous checkpoint or base)
    if checkpoint == 1:
        print(f"\nLoading base model {MODEL_NAME}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        prev_path = f"models/gemma-ckpt{checkpoint-1}-lora"
        if os.path.exists(prev_path):
            print(f"\nLoading previous checkpoint from {prev_path}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=prev_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            print(f"No previous checkpoint, loading base model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
    
    # Train
    output_dir = f"models/gemma-ckpt{checkpoint}-lora"
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,  # CRITICAL: Use 1 to prevent OOM during tokenization
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,  # Reduced from 4 to reduce memory
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            warmup_steps=30,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",
            # Memory optimization flags
            gradient_checkpointing=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            # Windows-specific: Disable torch.compile
            torch_compile=False,
            torch_compile_backend='inductor',
        ),
    )
    # Ensure dynamo is disabled before training (Unsloth may re-enable it)
    torch._dynamo.config.disable = True
    torch.compiler.disable()   # belt-and-suspenders for torch 2.7+

    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    # SECURITY FIX: Properly release GPU memory after training
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Update status
    record_ids = [r['id'] for r in records]
    update_status(supabase, record_ids, 'output_tuned')
    print(f"✅ Status updated to 'output_tuned' - ready for tuned generation")
    
    return output_dir

# =============================================================================
# STEP 4: output_tuned - Generate Tuned Student Output
# =============================================================================
def step_output_tuned(supabase, checkpoint):
    """Generate student_output_tuned using finetuned model"""
    print(f"\n{'='*60}")
    print(f"🤖 STEP: output_tuned - Generate Tuned Output for Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'output_tuned')
    
    # Validate minimum records
    if not validate_records_count(records, 'output_tuned'):
        return 0
    
    # Load finetuned model
    model_path = f"models/gemma-ckpt{checkpoint}-lora"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Run 'finetune' step first.")
        return 0
    
    print(f"\nLoading finetuned model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Generate outputs
    for item in tqdm(records, desc="Generating"):
        try:
            output, latency = generate_output(
                model, tokenizer,
                item['input'],
                item.get('context', '')
            )
            
            supabase.table('modelcomp_50k').update({
                'student_output_tuned': output[:5000],
                'latency_tuned': round(latency, 3),
                'status': 'score_tuned'
            }).eq('id', item['id']).execute()
            
        except Exception as e:
            print(f"Error on record {item['id']}: {e}")
            continue
    
    print(f"✅ Generated tuned outputs for {len(records)} records")
    print(f"✅ Status updated to 'score_tuned' - ready for final scoring")
    
    return len(records)

def get_worker_progress(supabase, checkpoint):
    """Get progress from all workers"""
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

    total = total_result.count or 0
    completed = completed_result.count or 0

    worker_counts = {}
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
            worker_counts[worker] = worker_counts.get(worker, 0) + 1

        if len(rows) < page_size:
            break
        offset += page_size

    pending = max(0, total - completed)
    return {
        'total': total,
        'completed': completed,
        'pending': pending,
        'workers': worker_counts,
        'progress_pct': (completed / total * 100) if total > 0 else 0
    }

def estimate_time_remaining(checkpoint, total_records, completed, avg_time_per_record=2.0):
    """Estimate time remaining assuming 2 parallel workers"""
    if completed >= total_records:
        return None, None
    
    pending = total_records - completed
    # With 2 workers in parallel, divide by 2
    effective_pending = pending / 2
    remaining_seconds = effective_pending * avg_time_per_record
    completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
    
    return completion_time, remaining_seconds

def step_output_tuned_distributed(supabase, checkpoint):
    """Generate student_output_tuned using distributed workers on multiple machines"""
    print(f"\n{'='*70}")
    print(f"🤖 STEP: output_tuned - Generate Tuned Output (Distributed)")
    print(f"   Checkpoint {checkpoint}")
    print(f"{'='*70}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'output_tuned')
    
    # Validate minimum records
    if not validate_records_count(records, 'output_tuned'):
        return 0
    
    total_records = len(records)
    
    # Check model exists
    model_path = f"models/gemma-ckpt{checkpoint}-lora"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 0
    
    print(f"\n{'='*70}")
    print(f"DISTRIBUTED INFERENCE INSTRUCTIONS")
    print(f"{'='*70}")
    print(f"\nTotal records to process: {total_records}")
    print(f"\nRun workers on different machines:")
    print(f"\n  MACBOOK PRO:")
    print(f"    python 13_distributed_worker.py --checkpoint {checkpoint} --worker-id mac-1")
    print(f"\n  WINDOWS RTX (optional parallel worker):")
    print(f"    python 13_distributed_worker.py --checkpoint {checkpoint} --worker-id rtx-1")
    print(f"\n  Both workers will coordinate via Supabase and process in parallel.")
    print(f"\n{'='*70}\n")
    
    # Track progress loop
    start_time = datetime.now()
    check_interval = 10  # seconds between progress checks
    last_completed = 0
    record_times = []
    
    try:
        while True:
            progress = get_worker_progress(supabase, checkpoint)
            completed = progress['completed']
            pending = progress['pending']
            pct = progress['progress_pct']
            
            # Calculate speed from actual data
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0 and completed > 0:
                avg_time_per_record = elapsed / completed
            else:
                avg_time_per_record = 2.0  # default estimate
            
            # Estimate completion
            completion_info = estimate_time_remaining(checkpoint, total_records, completed, avg_time_per_record)
            completion_time, remaining_sec = completion_info if completion_info[0] else (None, None)
            
            # Progress dashboard
            print(f"\n{'─'*70}")
            print(f"PROGRESS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'─'*70}")
            print(f"Completed: {completed:4d} / {total_records:4d} ({pct:5.1f}%)")
            print(f"Pending:   {pending:4d}")
            
            # Visual progress bar
            bar_length = 50
            filled = int(bar_length * pct / 100)
            bar = 'x' * filled + '.' * (bar_length - filled)
            print(f"[{bar}]")
            
            # Worker stats
            if progress['workers']:
                print(f"\nWorker contributions:")
                for worker, count in progress['workers'].items():
                    if worker is None:  # Skip None workers
                        continue
                    pct_worker = (count / completed * 100) if completed > 0 else 0
                    print(f"  {worker:10s}: {count:4d} records ({pct_worker:5.1f}%)")
            
            # Time estimates
            if avg_time_per_record > 0:
                print(f"\nAvg time per record: {avg_time_per_record:.2f}s")
            if completion_time and remaining_sec:
                hours = int(remaining_sec // 3600)
                mins = int((remaining_sec % 3600) // 60)
                secs = int(remaining_sec % 60)
                if hours > 0:
                    time_str = f"{hours}h {mins}m {secs}s"
                elif mins > 0:
                    time_str = f"{mins}m {secs}s"
                else:
                    time_str = f"{secs}s"
                print(f"Est. completion: {completion_time.strftime('%H:%M:%S')} ({time_str} remaining)")
            
            print(f"{'─'*70}")
            
            # Check if done
            if pending == 0 and completed == total_records:
                print(f"\n✅ All {total_records} records generated!")
                break
            
            # Wait before next check
            if pending > 0:
                time.sleep(check_interval)
            else:
                time.sleep(5)
    
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Workers are still processing...")
        progress = get_worker_progress(supabase, checkpoint)
        print(f"Current progress: {progress['completed']} / {total_records}")
    
    # Final update
    final_progress = get_worker_progress(supabase, checkpoint)
    remaining = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .eq('checkpoint', checkpoint)\
        .eq('status', 'output_tuned')\
        .execute()
    
    if remaining.count == 0:
        print(f"\n✅ All outputs generated!")
        print(f"✅ Status: {final_progress['completed']} / {total_records} ready for scoring")
        return final_progress['completed']
    else:
        print(f"\nNote: {remaining.count} records still pending")
        return final_progress['completed']

# =============================================================================
# STEP 5: score_tuned - Score Tuned Student Output
# =============================================================================
def step_score_tuned(supabase, checkpoint):
    """Score student_output_tuned against teacher (sevenb) using full evaluation matrix"""
    print(f"\n{'='*60}")
    print(f"📊 STEP: score_tuned - Score Tuned Output for Checkpoint {checkpoint}")
    print(f"   Using comprehensive evaluation metrics...")
    print(f"{'='*60}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'score_tuned')
    
    # Validate minimum records
    if not validate_records_count(records, 'score_tuned'):
        return 0
    
    all_metrics = []
    for item in tqdm(records, desc="Scoring tuned with full matrix"):
        tuned_out = item.get('student_output_tuned', '')
        teacher_out = item.get('sevenb', '')
        instruction = item.get('input', '')
        context = item.get('context', '')
        task_label = item.get('task_label', 'general_qa')
        
        if tuned_out and teacher_out:
            metrics = calculate_metrics(
                prediction=tuned_out,
                reference=teacher_out,
                instruction=instruction,
                context=context,
                task_label=task_label
            )
            score_tuned = round(metrics['overall'], 4)
        else:
            metrics = None
            score_tuned = None
        
        # Update all scores in Supabase as individual columns with _tuned suffix
        update_data = {
            'score_tuned': score_tuned,
            'status': 'completed'
        }
        
        # Store ALL metrics as individual columns (after finetuning with _tuned suffix)
        if metrics:
            update_data['structured_correctness_tuned'] = round(metrics['structured_correctness'], 4)
            update_data['task_success_tuned'] = round(metrics['task_success'], 4)
            update_data['instruction_following_tuned'] = round(metrics['instruction_following'], 4)
            update_data['coverage_tuned'] = round(metrics['coverage'], 4)
            update_data['faithfulness_tuned'] = round(metrics['faithfulness'], 4)
            update_data['hallucination_tuned'] = round(metrics['hallucination'], 4)
            update_data['context_grounding_tuned'] = round(metrics['context_grounding'], 4)
            update_data['conciseness_tuned'] = round(metrics['conciseness'], 4)
            update_data['rouge1_tuned'] = round(metrics['rouge1'], 4)
            update_data['rougel_tuned'] = round(metrics['rougel'], 4)
            update_data['bleu_tuned'] = round(metrics['bleu'], 4)
        
        supabase.table('modelcomp_50k').update(update_data).eq('id', item['id']).execute()
        
        if metrics:
            all_metrics.append(metrics)
    
    # Print summary of all metrics
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"📊 TUNED STUDENT EVALUATION SUMMARY (Checkpoint {checkpoint})")
        print(f"{'='*60}")
        print(f"   Records evaluated: {len(all_metrics)}")
        print(f"   ─────────────────────────────────────")
        print(f"   Overall Score:         {sum(m['overall'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ─────────────────────────────────────")
        print(f"   Structured Correctness: {sum(m['structured_correctness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Task Success:           {sum(m['task_success'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Instruction Following:  {sum(m['instruction_following'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Coverage:               {sum(m['coverage'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Faithfulness:           {sum(m['faithfulness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Hallucination:          {sum(m['hallucination'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Context Grounding:      {sum(m['context_grounding'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   Conciseness:            {sum(m['conciseness'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ─────────────────────────────────────")
        print(f"   ROUGE-1:                {sum(m['rouge1'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   ROUGE-L:                {sum(m['rougel'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"   BLEU:                   {sum(m['bleu'] for m in all_metrics)/len(all_metrics):.4f}")
        print(f"{'='*60}")
    
    avg_score = sum(m['overall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    print(f"\n✅ Status updated to 'completed' - ready for final calculation")
    
    return avg_score

# =============================================================================
# STEP 6: completed - Calculate Final Improvement
# =============================================================================
def step_completed(supabase, checkpoint):
    """Calculate improvement score and generate report"""
    print(f"\n{'='*60}")
    print(f"✅ STEP: completed - Calculate Improvement for Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    records = fetch_records_by_status(supabase, checkpoint, 'completed')
    
    # Validate minimum records
    if not validate_records_count(records, 'completed'):
        return None
    
    # Calculate improvements
    improvements = []
    for item in tqdm(records, desc="Calculating improvement"):
        score_before = item.get('score') or 0
        score_after = item.get('score_tuned') or 0
        improvement = round(score_after - score_before, 4)
        
        supabase.table('modelcomp_50k').update({
            'improvement': improvement
        }).eq('id', item['id']).execute()
        
        if score_before and score_after:
            improvements.append({
                'before': score_before,
                'after': score_after,
                'improvement': improvement
            })
    
    # Summary
    if improvements:
        avg_before = sum(i['before'] for i in improvements) / len(improvements)
        avg_after = sum(i['after'] for i in improvements) / len(improvements)
        avg_improvement = sum(i['improvement'] for i in improvements) / len(improvements)
        pct_improvement = (avg_improvement / avg_before * 100) if avg_before else 0
        
        print(f"\n{'='*60}")
        print(f"📈 CHECKPOINT {checkpoint} RESULTS")
        print(f"{'='*60}")
        print(f"   Records processed: {len(improvements)}")
        print(f"   Avg Score Before:  {avg_before:.4f}")
        print(f"   Avg Score After:   {avg_after:.4f}")
        print(f"   Avg Improvement:   {avg_improvement:+.4f} ({pct_improvement:+.1f}%)")
        print(f"{'='*60}")
        
        # Save report
        report = {
            'checkpoint': checkpoint,
            'timestamp': datetime.now().isoformat(),
            'records': len(improvements),
            'score_before': avg_before,
            'score_after': avg_after,
            'improvement': avg_improvement,
            'improvement_pct': pct_improvement
        }
        
        os.makedirs('reports/incremental', exist_ok=True)
        report_path = f"reports/incremental/checkpoint_{checkpoint}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved: {report_path}")
        
        return report
    
    return None

# =============================================================================
# RUN ALL - Execute all steps for a checkpoint
# =============================================================================
def run_all_steps(supabase, checkpoint):
    """Run all steps for a checkpoint automatically"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING ALL STEPS FOR CHECKPOINT {checkpoint}")
    print(f"{'='*60}")
    
    results = {}
    
    for step in STATUS_FLOW:
        print(f"\n{'─'*60}")
        print(f"▶️  Starting step: {step}")
        print(f"{'─'*60}")
        
        if step == 'score':
            result = step_score(supabase, checkpoint)
        elif step == 'finetune':
            result = step_finetune(supabase, checkpoint)
        elif step == 'output_tuned':
            result = step_output_tuned(supabase, checkpoint)
        elif step == 'score_tuned':
            result = step_score_tuned(supabase, checkpoint)
        elif step == 'completed':
            result = step_completed(supabase, checkpoint)
        
        results[step] = result
        
        # Check if step failed (returned 0/None for steps that should return data)
        if result is None or (isinstance(result, (int, float)) and result == 0):
            print(f"\n❌ Step '{step}' did not complete successfully. Stopping.")
            break
    
    print(f"\n{'='*60}")
    print(f"📋 RUN ALL SUMMARY - Checkpoint {checkpoint}")
    print(f"{'='*60}")
    for step, result in results.items():
        status = '✅' if result else '❌'
        print(f"   {status} {step}: {result}")
    
    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Incremental Learning Pipeline')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint number (1-10)')
    parser.add_argument('--step', type=str, 
                       choices=STATUS_FLOW + ['status'],
                       help='Step to run: score, finetune, output_tuned, score_tuned, completed, status')
    parser.add_argument('--run-all', action='store_true', help='Run all steps for checkpoint')
    parser.add_argument('--init', action='store_true', help='Initialize checkpoint (set status to score)')
    args = parser.parse_args()
    
    if args.checkpoint < 1 or args.checkpoint > 10:
        print("Error: Checkpoint must be 1-10")
        sys.exit(1)
    
    # Validate arguments
    if not args.step and not args.run_all and not args.init:
        print("Error: Must specify --step, --run-all, or --init")
        parser.print_help()
        sys.exit(1)
    
    checkpoint = args.checkpoint
    
    print(f"\n{'='*60}")
    print(f"🚀 INCREMENTAL LEARNING - Checkpoint {checkpoint}")
    print(f"{'='*60}")
    
    supabase = get_supabase()
    
    # Handle --init
    if args.init:
        init_checkpoint(supabase, checkpoint)
        return
    
    # Handle --run-all
    if args.run_all:
        run_all_steps(supabase, checkpoint)
        return
    
    step = args.step
    
    # Run the appropriate step
    if step == 'status':
        step_status(supabase, checkpoint)
    elif step == 'score':
        step_score(supabase, checkpoint)
    elif step == 'finetune':
        step_finetune(supabase, checkpoint)
    elif step == 'output_tuned':
        step_output_tuned_distributed(supabase, checkpoint)
    elif step == 'score_tuned':
        step_score_tuned(supabase, checkpoint)
    elif step == 'completed':
        step_completed(supabase, checkpoint)
    
    # Show next step (only for non-status steps)
    if step != 'status' and step in STATUS_FLOW:
        current_idx = STATUS_FLOW.index(step)
        if current_idx < len(STATUS_FLOW) - 1:
            next_step = STATUS_FLOW[current_idx + 1]
            print(f"\n   Next: python experiment/12_train_incremental.py --checkpoint {checkpoint} --step {next_step}")
        else:
            print(f"\n   ✅ Checkpoint {checkpoint} complete!")
            if checkpoint < 10:
                print(f"   Next checkpoint: python experiment/12_train_incremental.py --checkpoint {checkpoint + 1} --init")

if __name__ == "__main__":
    main()
