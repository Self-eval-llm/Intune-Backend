#!/usr/bin/env python3
"""
Batch Training Pipeline - Status-Based Workflow
================================================
Train on ALL scored data at once (no checkpoint filtering).
For comparison with Incremental Learning approach.

Status Flow (Pure Status-Based):
  score -> finetune -> output_tuned -> score_tuned -> completed

Unlike incremental (which trains checkpoint by checkpoint),
batch mode trains on ALL available scored records at once.

Usage:
    # Check status
    python experiment/phase2_incremental/13_train_batch.py --status

    # Run individual steps
    python experiment/phase2_incremental/13_train_batch.py --step finetune
    python experiment/phase2_incremental/13_train_batch.py --step output_tuned
    python experiment/phase2_incremental/13_train_batch.py --step score_tuned
    python experiment/phase2_incremental/13_train_batch.py --step completed

    # Run all steps
    python experiment/phase2_incremental/13_train_batch.py --run-all
"""

import os
import sys
import json
import time
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Disable torch dynamo for Windows compatibility
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Training imports
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Evaluation imports - use same method as 12_train_incremental.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
try:
    from importlib import import_module
    eval_metrics = import_module('06_eval_metrics')
    evaluate_single_output = eval_metrics.evaluate_single_output
except Exception as e:
    print(f"Warning: Could not import eval_metrics: {e}")
    evaluate_single_output = None

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Config
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512
MIN_RECORDS = 100  # Minimum records to proceed

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Output paths
BATCH_MODEL_PATH = "models/gemma-batch-lora"
BATCH_REPORT_PATH = "reports/batch_learning"


def get_supabase():
    return create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))


def fetch_records_by_status(supabase, status, limit=None):
    """Fetch ALL records with given status (no checkpoint filter)"""
    print(f"\nFetching all records with status='{status}'...")

    all_records = []
    offset = 0
    batch_size = 1000

    while True:
        query = supabase.table('modelcomp_50k')\
            .select('*')\
            .eq('status', status)\
            .range(offset, offset + batch_size - 1)

        result = query.execute()

        if not result.data:
            break

        all_records.extend(result.data)
        offset += batch_size

        if limit and len(all_records) >= limit:
            all_records = all_records[:limit]
            break

        if len(result.data) < batch_size:
            break

    print(f"Fetched {len(all_records)} records with status='{status}'")
    return all_records


def count_by_status(supabase):
    """Count records at each status"""
    counts = {}
    for status in ['score', 'finetune', 'output_tuned', 'score_tuned', 'completed']:
        result = supabase.table('modelcomp_50k')\
            .select('id', count='exact')\
            .eq('status', status)\
            .execute()
        counts[status] = result.count

    # NULL status
    result = supabase.table('modelcomp_50k')\
        .select('id', count='exact')\
        .is_('status', 'null')\
        .execute()
    counts['null'] = result.count

    return counts


def update_status(supabase, record_ids, new_status):
    """Update status for multiple records"""
    print(f"Updating {len(record_ids)} records to status='{new_status}'...")

    batch_size = 100
    for i in range(0, len(record_ids), batch_size):
        batch = record_ids[i:i + batch_size]
        for rid in batch:
            try:
                supabase.table('modelcomp_50k')\
                    .update({'status': new_status})\
                    .eq('id', rid)\
                    .execute()
            except Exception as e:
                print(f"Error updating {rid}: {e}")


def step_status(supabase):
    """Show current status distribution"""
    print("\n" + "="*60)
    print("BATCH MODE - STATUS OVERVIEW")
    print("="*60)

    counts = count_by_status(supabase)
    total = sum(counts.values())

    print(f"\nTotal records: {total:,}")
    print("-"*40)
    print(f"  NULL (not started):   {counts.get('null', 0):,}")
    print(f"  score:                {counts.get('score', 0):,}")
    print(f"  finetune:             {counts.get('finetune', 0):,}")
    print(f"  output_tuned:         {counts.get('output_tuned', 0):,}")
    print(f"  score_tuned:          {counts.get('score_tuned', 0):,}")
    print(f"  completed:            {counts.get('completed', 0):,}")
    print("-"*40)

    # Recommendations
    if counts.get('score', 0) >= MIN_RECORDS:
        print(f"\n[READY] {counts['score']:,} records ready for finetune")
        print("   Run: python experiment/phase2_incremental/13_train_batch.py --step finetune")
    elif counts.get('finetune', 0) >= MIN_RECORDS:
        print(f"\n[READY] {counts['finetune']:,} records ready for output_tuned")
        print("   Run: python experiment/phase2_incremental/13_train_batch.py --step output_tuned")
    elif counts.get('output_tuned', 0) >= MIN_RECORDS:
        print(f"\n[READY] {counts['output_tuned']:,} records ready for score_tuned")
        print("   Run: python experiment/phase2_incremental/13_train_batch.py --step score_tuned")
    elif counts.get('score_tuned', 0) >= MIN_RECORDS:
        print(f"\n[READY] {counts['score_tuned']:,} records ready for completed")
        print("   Run: python experiment/phase2_incremental/13_train_batch.py --step completed")

    print("="*60)
    return counts


def fetch_records_ready_to_score(supabase, limit=None):
    """Fetch ALL records with student_output but NULL status (ready to score)"""
    print(f"\nFetching records ready to score (has student_output, status=NULL)...")

    all_records = []
    offset = 0
    batch_size = 1000

    while True:
        query = supabase.table('modelcomp_50k')\
            .select('*')\
            .is_('status', 'null')\
            .neq('student_output', '')\
            .not_.is_('student_output', 'null')\
            .range(offset, offset + batch_size - 1)

        result = query.execute()

        if not result.data:
            break

        all_records.extend(result.data)
        offset += batch_size

        if limit and len(all_records) >= limit:
            all_records = all_records[:limit]
            break

        if len(result.data) < batch_size:
            break

    print(f"Fetched {len(all_records)} records ready to score")
    return all_records


def step_score(supabase):
    """Score ALL available student outputs (no checkpoint filter)"""
    print("\n" + "="*60)
    print("BATCH SCORE - Scoring ALL student outputs")
    print("="*60)

    records = fetch_records_ready_to_score(supabase)

    if len(records) < MIN_RECORDS:
        print(f"Not enough records ({len(records)}) - need at least {MIN_RECORDS}")
        return False

    print(f"\nScoring {len(records):,} student outputs")

    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    success_count = 0
    for record in tqdm(records, desc="Scoring base outputs"):
        try:
            instruction = record.get('input', '')
            teacher_out = record.get('sevenb', '')
            student_out = record.get('student_output', '')
            context = record.get('context', '') or ''
            task_label = record.get('task_label', 'general_qa')

            if not student_out or not teacher_out:
                continue

            # Calculate metrics
            if evaluate_single_output:
                try:
                    metrics = evaluate_single_output(
                        instruction=instruction,
                        student_output=student_out,
                        teacher_output=teacher_out,
                        context=context,
                        task_label=task_label
                    )
                except Exception as e:
                    metrics = {
                        'structured_correctness': 0.5,
                        'task_success': 0.5,
                        'instruction_following': 0.5,
                        'coverage': 0.5,
                        'faithfulness': 0.5,
                        'hallucination': 0.5,
                        'context_grounding': 0.5,
                        'overall_score': 0.5,
                        'conciseness': 0.5
                    }
            else:
                metrics = {
                    'structured_correctness': 0.5,
                    'task_success': 0.5,
                    'instruction_following': 0.5,
                    'coverage': 0.5,
                    'faithfulness': 0.5,
                    'hallucination': 0.5,
                    'context_grounding': 0.5,
                    'overall_score': 0.5,
                    'conciseness': 0.5
                }

            # ROUGE/BLEU
            rouge_scores = rouge.score(teacher_out, student_out)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rougel = rouge_scores['rougeL'].fmeasure

            try:
                bleu = sentence_bleu(
                    [teacher_out.split()],
                    student_out.split(),
                    smoothing_function=smooth
                )
            except:
                bleu = 0.0

            # Update database
            supabase.table('modelcomp_50k')\
                .update({
                    'score': metrics.get('overall_score', 0.5),
                    'structured_correctness': metrics.get('structured_correctness', 0.5),
                    'task_success': metrics.get('task_success', 0.5),
                    'instruction_following': metrics.get('instruction_following', 0.5),
                    'coverage': metrics.get('coverage', 0.5),
                    'faithfulness': metrics.get('faithfulness', 0.5),
                    'hallucination': metrics.get('hallucination', 0.5),
                    'context_grounding': metrics.get('context_grounding', 0.5),
                    'conciseness': metrics.get('conciseness', 0.5),
                    'rouge1': rouge1,
                    'rougel': rougel,
                    'bleu': bleu,
                    'status': 'score'
                })\
                .eq('id', record['id'])\
                .execute()

            success_count += 1

        except Exception as e:
            print(f"\nError for {record['id']}: {e}")

    print(f"\n[SUCCESS] Scored {success_count}/{len(records)} student outputs")
    print(f"Records now have status='score' and are ready for finetune")
    return True


def step_finetune(supabase):
    """Fine-tune on ALL records with status='score'"""
    print("\n" + "="*60)
    print("BATCH FINETUNE - Training on ALL scored data")
    print("="*60)

    records = fetch_records_by_status(supabase, 'score')

    if len(records) < MIN_RECORDS:
        print(f"Not enough records ({len(records)}) - need at least {MIN_RECORDS}")
        return False

    print(f"\nTraining on {len(records):,} records (ALL scored data)")

    # Prepare dataset
    print("\nPreparing training dataset...")
    training_data = []
    for item in tqdm(records, desc="Formatting"):
        context = item.get('context', '') or ''
        if context:
            text = f"### Instruction:\n{item['input']}\n\n### Context:\n{context}\n\n### Response:\n{item['sevenb']}"
        else:
            text = f"### Instruction:\n{item['input']}\n\n### Response:\n{item['sevenb']}"
        training_data.append({"text": text})

    dataset = Dataset.from_list(training_data)
    print(f"Dataset size: {len(dataset)}")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
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
    os.makedirs(BATCH_MODEL_PATH, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=BATCH_MODEL_PATH,
            save_strategy="epoch",
        ),
    )

    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    # Save model
    model.save_pretrained(BATCH_MODEL_PATH)
    tokenizer.save_pretrained(BATCH_MODEL_PATH)
    print(f"\nModel saved to {BATCH_MODEL_PATH}")
    print(f"Training time: {train_time/60:.1f} minutes")

    # Update status
    record_ids = [r['id'] for r in records]
    update_status(supabase, record_ids, 'finetune')

    # Save report
    os.makedirs(BATCH_REPORT_PATH, exist_ok=True)
    report = {
        "step": "finetune",
        "records_trained": len(records),
        "train_time_minutes": train_time / 60,
        "model_path": BATCH_MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }
    with open(f"{BATCH_REPORT_PATH}/finetune_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("\n[SUCCESS] Batch finetune completed!")
    return True


def step_output_tuned(supabase):
    """Generate tuned outputs for ALL records with status='finetune'"""
    print("\n" + "="*60)
    print("BATCH OUTPUT_TUNED - Generating tuned outputs")
    print("="*60)

    records = fetch_records_by_status(supabase, 'finetune')

    if len(records) < MIN_RECORDS:
        print(f"Not enough records ({len(records)}) - need at least {MIN_RECORDS}")
        return False

    print(f"\nGenerating outputs for {len(records):,} records")

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model from {BATCH_MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BATCH_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Generate outputs
    success_count = 0
    for record in tqdm(records, desc="Generating"):
        try:
            context = record.get('context', '') or ''
            if context:
                prompt = f"### Instruction:\n{record['input']}\n\n### Context:\n{context}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{record['input']}\n\n### Response:\n"

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

            # Update database
            supabase.table('modelcomp_50k')\
                .update({
                    'student_output_tuned': response,
                    'latency_tuned': latency,
                    'status': 'output_tuned'
                })\
                .eq('id', record['id'])\
                .execute()

            success_count += 1

        except Exception as e:
            print(f"\nError for {record['id']}: {e}")

    print(f"\n[SUCCESS] Generated {success_count}/{len(records)} tuned outputs")
    return True


def step_score_tuned(supabase):
    """Score tuned outputs for ALL records with status='output_tuned'"""
    print("\n" + "="*60)
    print("BATCH SCORE_TUNED - Scoring tuned outputs")
    print("="*60)

    records = fetch_records_by_status(supabase, 'output_tuned')

    if len(records) < MIN_RECORDS:
        print(f"Not enough records ({len(records)}) - need at least {MIN_RECORDS}")
        return False

    print(f"\nScoring {len(records):,} records")

    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    success_count = 0
    for record in tqdm(records, desc="Scoring"):
        try:
            instruction = record.get('input', '')
            teacher_out = record.get('sevenb', '')
            student_out = record.get('student_output_tuned', '')
            context = record.get('context', '') or ''
            task_label = record.get('task_label', 'general_qa')

            if not student_out:
                continue

            # Calculate metrics using eval_metrics module
            try:
                metrics = eval_metrics.evaluate_single_output(
                    instruction=instruction,
                    student_output=student_out,
                    teacher_output=teacher_out,
                    context=context,
                    task_label=task_label
                )
            except:
                # Fallback to basic metrics
                metrics = {
                    'structured_correctness': 0.5,
                    'task_success': 0.5,
                    'instruction_following': 0.5,
                    'coverage': 0.5,
                    'faithfulness': 0.5,
                    'hallucination': 0.5,
                    'context_grounding': 0.5,
                    'overall_score': 0.5,
                    'conciseness': 0.5
                }

            # ROUGE/BLEU
            rouge_scores = rouge.score(teacher_out, student_out)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rougel = rouge_scores['rougeL'].fmeasure

            try:
                bleu = sentence_bleu(
                    [teacher_out.split()],
                    student_out.split(),
                    smoothing_function=smooth
                )
            except:
                bleu = 0.0

            # Update database
            supabase.table('modelcomp_50k')\
                .update({
                    'score_tuned': metrics.get('overall_score', 0.5),
                    'structured_correctness_tuned': metrics.get('structured_correctness', 0.5),
                    'task_success_tuned': metrics.get('task_success', 0.5),
                    'instruction_following_tuned': metrics.get('instruction_following', 0.5),
                    'coverage_tuned': metrics.get('coverage', 0.5),
                    'faithfulness_tuned': metrics.get('faithfulness', 0.5),
                    'hallucination_tuned': metrics.get('hallucination', 0.5),
                    'context_grounding_tuned': metrics.get('context_grounding', 0.5),
                    'conciseness_tuned': metrics.get('conciseness', 0.5),
                    'rouge1_tuned': rouge1,
                    'rougel_tuned': rougel,
                    'bleu_tuned': bleu,
                    'status': 'score_tuned'
                })\
                .eq('id', record['id'])\
                .execute()

            success_count += 1

        except Exception as e:
            print(f"\nError for {record['id']}: {e}")

    print(f"\n[SUCCESS] Scored {success_count}/{len(records)} records")
    return True


def step_completed(supabase):
    """Calculate improvements and mark as completed"""
    print("\n" + "="*60)
    print("BATCH COMPLETED - Final analysis")
    print("="*60)

    records = fetch_records_by_status(supabase, 'score_tuned')

    if len(records) < MIN_RECORDS:
        print(f"Not enough records ({len(records)}) - need at least {MIN_RECORDS}")
        return False

    print(f"\nFinalizing {len(records):,} records")

    improvements = []
    for record in tqdm(records, desc="Calculating improvements"):
        try:
            base_score = record.get('score', 0) or 0
            tuned_score = record.get('score_tuned', 0) or 0
            improvement = tuned_score - base_score

            supabase.table('modelcomp_50k')\
                .update({
                    'improvement': improvement,
                    'status': 'completed'
                })\
                .eq('id', record['id'])\
                .execute()

            improvements.append(improvement)

        except Exception as e:
            print(f"\nError for {record['id']}: {e}")

    # Summary
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        positive = sum(1 for i in improvements if i > 0)
        negative = sum(1 for i in improvements if i < 0)

        print("\n" + "="*60)
        print("BATCH TRAINING RESULTS")
        print("="*60)
        print(f"Total records: {len(improvements):,}")
        print(f"Average improvement: {avg_improvement:.4f}")
        print(f"Improved: {positive:,} ({100*positive/len(improvements):.1f}%)")
        print(f"Degraded: {negative:,} ({100*negative/len(improvements):.1f}%)")
        print("="*60)

        # Save report
        os.makedirs(BATCH_REPORT_PATH, exist_ok=True)
        report = {
            "mode": "batch",
            "total_records": len(improvements),
            "avg_improvement": avg_improvement,
            "improved_count": positive,
            "degraded_count": negative,
            "timestamp": datetime.now().isoformat()
        }
        with open(f"{BATCH_REPORT_PATH}/batch_results.json", 'w') as f:
            json.dump(report, f, indent=2)

    print("\n[SUCCESS] Batch training pipeline completed!")
    return True


def run_all_steps(supabase):
    """Run all steps in sequence"""
    print("\n" + "="*60)
    print("BATCH MODE - RUNNING ALL STEPS")
    print("="*60)

    steps = [
        ("score", step_score),
        ("finetune", step_finetune),
        ("output_tuned", step_output_tuned),
        ("score_tuned", step_score_tuned),
        ("completed", step_completed),
    ]

    for step_name, step_fn in steps:
        print(f"\n>>> Running step: {step_name}")
        success = step_fn(supabase)
        if not success:
            print(f"\nStopped at step: {step_name}")
            break

    print("\n" + "="*60)
    print("BATCH PIPELINE FINISHED")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Batch Training Pipeline (Status-Based)')
    parser.add_argument('--status', action='store_true', help='Show status overview')
    parser.add_argument('--step', type=str,
                       choices=['score', 'finetune', 'output_tuned', 'score_tuned', 'completed'],
                       help='Step to run')
    parser.add_argument('--run-all', action='store_true', help='Run all steps')
    args = parser.parse_args()

    supabase = get_supabase()

    if args.status or (not args.step and not args.run_all):
        step_status(supabase)
        return

    if args.run_all:
        run_all_steps(supabase)
        return

    step_functions = {
        'score': step_score,
        'finetune': step_finetune,
        'output_tuned': step_output_tuned,
        'score_tuned': step_score_tuned,
        'completed': step_completed,
    }

    if args.step in step_functions:
        step_functions[args.step](supabase)


if __name__ == "__main__":
    main()
