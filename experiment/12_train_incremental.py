"""
Stage-by-Stage Incremental Learning with Evaluation
====================================================
For each stage (1-10):
1. Finetune on cumulative data (Stage 1=5K, Stage 2=10K, etc.)
2. Generate outputs with finetuned model
3. Evaluate against teacher (sevenb) and compare with previous checkpoint
4. Update Supabase with student_output_ckptN, score_ckptN, latency_ckptN
5. STOP - run next stage manually

Usage:
    python experiment/12_stage_incremental.py --stage 1
    python experiment/12_stage_incremental.py --stage 2
    ...
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

# Training imports
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Evaluation imports
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Config
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048
RECORDS_PER_STAGE = 5000
EVAL_SAMPLE_SIZE = 500  # Evaluate on 500 samples per stage
MAX_NEW_TOKENS = 512

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

def get_supabase():
    """Get Supabase client"""
    return create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def fetch_training_data(supabase, stage):
    """Fetch cumulative training data up to this stage"""
    # Stage 1 = checkpoints 1 (5K)
    # Stage 2 = checkpoints 1-2 (10K)
    # etc.
    print(f"\nFetching training data for Stage {stage} (checkpoints 1-{stage})...")
    
    all_data = []
    for ckpt in range(1, stage + 1):
        result = supabase.table('modelcomp_50k')\
            .select('id, input, context, sevenb')\
            .eq('checkpoint', ckpt)\
            .execute()
        all_data.extend(result.data)
    
    print(f"Fetched {len(all_data)} training records")
    return all_data

def fetch_eval_data(supabase, stage):
    """Fetch evaluation data - use records from current stage"""
    print(f"\nFetching evaluation data from Stage {stage}...")
    
    result = supabase.table('modelcomp_50k')\
        .select('id, input, context, sevenb, student_output')\
        .eq('checkpoint', stage)\
        .limit(EVAL_SAMPLE_SIZE)\
        .execute()
    
    print(f"Fetched {len(result.data)} evaluation records")
    return result.data

def format_for_training(data):
    """Format data for SFT training"""
    formatted = []
    for item in data:
        context = item.get('context') or ''
        if context:
            text = f"### Instruction:\n{item['input']}\n\n### Context:\n{context}\n\n### Response:\n{item['sevenb']}"
        else:
            text = f"### Instruction:\n{item['input']}\n\n### Response:\n{item['sevenb']}"
        formatted.append({"text": text})
    return Dataset.from_list(formatted)

def load_base_model():
    """Load base model with LoRA"""
    print(f"\nLoading {MODEL_NAME}...")
    
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
    
    return model, tokenizer

def load_previous_checkpoint(stage):
    """Load model from previous checkpoint if exists"""
    if stage == 1:
        return load_base_model()
    
    prev_path = f"models/gemma-stage{stage-1}-lora"
    if os.path.exists(prev_path):
        print(f"\nLoading previous checkpoint from {prev_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=prev_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        return model, tokenizer
    else:
        print(f"No previous checkpoint found, starting from base model")
        return load_base_model()

def train_stage(model, tokenizer, dataset, stage):
    """Train model on stage data"""
    output_dir = f"models/gemma-stage{stage}-lora"
    
    print(f"\nTraining Stage {stage} on {len(dataset)} records...")
    print(f"Output: {output_dir}")
    
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
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",
        ),
    )
    
    trainer.train()
    
    # Save LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Stage {stage} training complete! Saved to {output_dir}")
    return model, tokenizer

def generate_output(model, tokenizer, instruction, context=""):
    """Generate output with finetuned model"""
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

def calculate_metrics(prediction, reference):
    """Calculate evaluation metrics"""
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    
    # BLEU
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    except:
        bleu = 0.0
    
    # Overall score (weighted average)
    overall = (
        rouge_scores['rouge1'].fmeasure * 0.3 +
        rouge_scores['rougeL'].fmeasure * 0.4 +
        bleu * 0.3
    )
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu,
        'overall': overall
    }

def generate_and_evaluate(model, tokenizer, eval_data, stage):
    """Generate outputs and evaluate"""
    print(f"\nGenerating and evaluating on {len(eval_data)} samples...")
    
    FastLanguageModel.for_inference(model)
    
    results = []
    all_scores = []
    
    for item in tqdm(eval_data, desc="Evaluating"):
        try:
            output, latency = generate_output(
                model, tokenizer,
                item['input'],
                item.get('context', '')
            )
            
            # Calculate metrics against teacher (sevenb)
            metrics = calculate_metrics(output, item['sevenb'])
            
            results.append({
                'id': item['id'],
                'output': output[:5000],
                'latency': round(latency, 3),
                'score': round(metrics['overall'], 4),
                'metrics': metrics
            })
            
            all_scores.append(metrics['overall'])
            
        except Exception as e:
            print(f"Error on record {item['id']}: {e}")
            continue
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n📊 Stage {stage} Average Score: {avg_score:.4f}")
    
    return results, avg_score

def update_supabase(supabase, results, stage):
    """Update Supabase with stage results"""
    output_col = f"student_output_ckpt{stage}"
    score_col = f"score_ckpt{stage}"
    latency_col = f"latency_ckpt{stage}"
    
    print(f"\nUpdating Supabase: {output_col}, {score_col}, {latency_col}...")
    
    for result in tqdm(results, desc="Updating"):
        supabase.table('modelcomp_50k').update({
            output_col: result['output'],
            score_col: result['score'],
            latency_col: result['latency']
        }).eq('id', result['id']).execute()
    
    print(f"✅ Updated {len(results)} records")

def compare_with_previous(supabase, stage, current_score):
    """Compare current stage score with previous"""
    print(f"\n📈 Stage {stage} Comparison:")
    print(f"   Current score: {current_score:.4f}")
    
    if stage == 1:
        # Compare with base student_output
        result = supabase.table('modelcomp_50k')\
            .select('student_output, sevenb')\
            .eq('checkpoint', 1)\
            .limit(EVAL_SAMPLE_SIZE)\
            .execute()
        
        if result.data and result.data[0].get('student_output'):
            base_scores = []
            for item in result.data:
                if item.get('student_output') and item.get('sevenb'):
                    metrics = calculate_metrics(item['student_output'], item['sevenb'])
                    base_scores.append(metrics['overall'])
            
            if base_scores:
                base_avg = sum(base_scores) / len(base_scores)
                improvement = current_score - base_avg
                print(f"   Base model score: {base_avg:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement/base_avg*100:+.1f}%)")
                return {'base': base_avg, 'current': current_score, 'improvement': improvement}
    else:
        # Compare with previous checkpoint
        prev_col = f"score_ckpt{stage-1}"
        result = supabase.table('modelcomp_50k')\
            .select(prev_col)\
            .eq('checkpoint', stage)\
            .not_.is_(prev_col, 'null')\
            .limit(EVAL_SAMPLE_SIZE)\
            .execute()
        
        if result.data:
            prev_scores = [r[prev_col] for r in result.data if r.get(prev_col)]
            if prev_scores:
                prev_avg = sum(prev_scores) / len(prev_scores)
                improvement = current_score - prev_avg
                print(f"   Previous stage score: {prev_avg:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement/prev_avg*100:+.1f}%)")
                return {'previous': prev_avg, 'current': current_score, 'improvement': improvement}
    
    return {'current': current_score}

def save_stage_report(stage, avg_score, comparison, eval_results):
    """Save stage report to file"""
    report = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'training_records': stage * RECORDS_PER_STAGE,
        'eval_samples': len(eval_results),
        'average_score': avg_score,
        'comparison': comparison,
        'sample_results': eval_results[:5]  # Save first 5 as samples
    }
    
    report_path = f"reports/stage{stage}_report.json"
    os.makedirs('reports', exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Run incremental learning stage')
    parser.add_argument('--stage', type=int, required=True, help='Stage number (1-10)')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, only evaluate')
    args = parser.parse_args()
    
    if args.stage < 1 or args.stage > 10:
        print("Error: Stage must be 1-10")
        sys.exit(1)
    
    stage = args.stage
    print(f"\n{'='*60}")
    print(f"🚀 STAGE {stage} INCREMENTAL LEARNING")
    print(f"   Training on: {stage * RECORDS_PER_STAGE} cumulative records")
    print(f"   Evaluating on: {EVAL_SAMPLE_SIZE} samples")
    print(f"{'='*60}")
    
    supabase = get_supabase()
    
    # Step 1: Fetch data
    train_data = fetch_training_data(supabase, stage)
    eval_data = fetch_eval_data(supabase, stage)
    
    # Step 2: Load model
    if args.skip_train and os.path.exists(f"models/gemma-stage{stage}-lora"):
        print(f"\nLoading existing Stage {stage} model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            f"models/gemma-stage{stage}-lora",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        # Load previous checkpoint or base model
        model, tokenizer = load_previous_checkpoint(stage)
        
        # Step 3: Train
        dataset = format_for_training(train_data)
        model, tokenizer = train_stage(model, tokenizer, dataset, stage)
    
    # Step 4: Generate and evaluate
    eval_results, avg_score = generate_and_evaluate(model, tokenizer, eval_data, stage)
    
    # Step 5: Update Supabase
    update_supabase(supabase, eval_results, stage)
    
    # Step 6: Compare with previous
    comparison = compare_with_previous(supabase, stage, avg_score)
    
    # Step 7: Save report
    save_stage_report(stage, avg_score, comparison, eval_results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ STAGE {stage} COMPLETE!")
    print(f"   Score: {avg_score:.4f}")
    if comparison.get('improvement'):
        print(f"   Improvement: {comparison['improvement']:+.4f}")
    print(f"\n   Next: python experiment/12_stage_incremental.py --stage {stage+1}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
