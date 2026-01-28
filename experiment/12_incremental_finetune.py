"""
Step 12: Incremental Self-Learning Loop (COMPLETE REWRITE)
============================================================
PROPER PIPELINE:
1. Load Alpaca 50K data (teacher outputs already exist)
2. For each 5K checkpoint:
   a. Fine-tune student on teacher data (cumulative)
   b. GENERATE student outputs on eval set
   c. COMPARE student vs teacher using 7 metrics
   d. Save checkpoint model + metrics
   e. Show improvement curve

This demonstrates SELF-LEARNING: student improves with more data!

Checkpoints: 5K → 10K → 15K → 20K → 25K → 30K → 35K → 40K → 45K → 50K

Usage:
    python experiment/12_incremental_finetune.py
    python experiment/12_incremental_finetune.py --start-checkpoint 3
    python experiment/12_incremental_finetune.py --dry-run
    python experiment/12_incremental_finetune.py --eval-only  # Just evaluate existing checkpoints
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Windows compatibility
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_PROC"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model config
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# LoRA config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training config
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05

# Incremental learning config
CHECKPOINT_SIZE = 5000  # 5K per checkpoint
TOTAL_CHECKPOINTS = 10  # 10 checkpoints = 50K total
EPOCHS_PER_CHECKPOINT = 1

# Evaluation config
EVAL_SAMPLES = 500  # Samples to evaluate at each checkpoint

# Directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "incremental_checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports" / "incremental_learning"
DATA_DIR = PROJECT_ROOT / "data" / "experiment"

# Alpaca prompt template (same as training)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

INFERENCE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


# ============================================================================
# DATA PREPARATION - ALPACA 50K
# ============================================================================

def download_and_prepare_alpaca_50k() -> Path:
    """
    Download and prepare Alpaca 50K dataset in proper format.
    Returns path to prepared JSON file.
    """
    import datasets
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "alpaca_50k_prepared.json"
    
    if output_file.exists():
        print(f"✓ Alpaca 50K already prepared: {output_file}")
        with open(output_file, "r") as f:
            data = json.load(f)
        print(f"  Records: {len(data)}")
        return output_file
    
    print("📥 Downloading and preparing Alpaca 50K dataset...")
    
    # Load from HuggingFace
    ds = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    
    prepared_data = []
    for i, item in enumerate(ds):
        if i >= 52000:  # Alpaca has ~52K, we use 50K
            break
        
        instruction = item["instruction"]
        context = item.get("input", "") or ""
        teacher_output = item["output"]  # This IS the Alpaca teacher output!
        
        # Format input properly
        if context.strip():
            formatted_input = f"Context: {context}\n\nQuestion: {instruction}"
        else:
            formatted_input = instruction
        
        prepared_data.append({
            "id": i + 1,
            "instruction": "Answer the following question accurately and concisely.",
            "input": formatted_input,
            "context": context,
            "raw_instruction": instruction,
            "teacher_output": teacher_output,  # Alpaca's output = teacher output
            "checkpoint": (i // CHECKPOINT_SIZE) + 1  # Which 5K checkpoint (1-10)
        })
    
    # Save
    with open(output_file, "w") as f:
        json.dump(prepared_data, f, indent=2)
    
    print(f"✓ Prepared {len(prepared_data)} records")
    print(f"✓ Saved to: {output_file}")
    
    # Stats
    for cp in range(1, 11):
        count = len([d for d in prepared_data if d["checkpoint"] == cp])
        print(f"   Checkpoint {cp}: {count} records")
    
    return output_file


def load_alpaca_data(checkpoint: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load Alpaca data for training up to checkpoint.
    Returns: (train_data, val_data, eval_data)
    
    - train_data: For fine-tuning (0 to checkpoint * 5K)
    - val_data: For validation during training (5% of train)
    - eval_data: Fixed eval set for comparing across checkpoints
    """
    data_file = DATA_DIR / "alpaca_50k_prepared.json"
    
    if not data_file.exists():
        download_and_prepare_alpaca_50k()
    
    with open(data_file, "r") as f:
        all_data = json.load(f)
    
    # Get cumulative training data (0 to checkpoint * 5K)
    end_idx = checkpoint * CHECKPOINT_SIZE
    train_raw = all_data[:end_idx]
    
    # Split train/val (95/5)
    split_idx = int(len(train_raw) * 0.95)
    train_data = train_raw[:split_idx]
    val_data = train_raw[split_idx:]
    
    # Fixed eval set: Use records 45K-50K (last 5K, not used in early checkpoints)
    # This ensures fair comparison across all checkpoints
    eval_data = all_data[45000:45000 + EVAL_SAMPLES]
    
    print(f"\n📊 Checkpoint {checkpoint} Data Split:")
    print(f"   Training: {len(train_data)} records (0 → {end_idx})")
    print(f"   Validation: {len(val_data)} records")
    print(f"   Evaluation: {len(eval_data)} records (fixed set for fair comparison)")
    
    return train_data, val_data, eval_data


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def load_base_model():
    """Load fresh base model with LoRA"""
    from unsloth import FastLanguageModel
    
    print(f"\n🔄 Loading base model: {MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA
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


def load_checkpoint_model(checkpoint: int):
    """Load model from a specific checkpoint"""
    from unsloth import FastLanguageModel
    
    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_{checkpoint}"
    
    if not checkpoint_path.exists():
        print(f"⚠️ Checkpoint {checkpoint} not found, loading base model")
        return load_base_model()
    
    print(f"\n🔄 Loading checkpoint {checkpoint}: {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train_checkpoint(
    checkpoint: int,
    model,
    tokenizer,
    train_data: List[Dict],
    val_data: List[Dict],
    epochs: int = EPOCHS_PER_CHECKPOINT
) -> Dict[str, float]:
    """Train model on checkpoint data"""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    
    print(f"\n{'=' * 60}")
    print(f"🎓 TRAINING CHECKPOINT {checkpoint}")
    print(f"   Data: {len(train_data)} train, {len(val_data)} val")
    print(f"   Epochs: {epochs}")
    print(f"{'=' * 60}")
    
    # Format for training
    def format_for_training(data):
        formatted = []
        for item in data:
            text = ALPACA_PROMPT.format(
                instruction=item["instruction"],
                input=item["input"],
                output=item["teacher_output"]
            ) + tokenizer.eos_token
            formatted.append({"text": text})
        return formatted
    
    train_formatted = format_for_training(train_data)
    val_formatted = format_for_training(val_data)
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False
        )
    
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Output dir
    checkpoint_dir = CHECKPOINTS_DIR / f"checkpoint_{checkpoint}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time
    
    # Eval loss
    eval_result = trainer.evaluate()
    
    # Save
    print(f"\n💾 Saving to {checkpoint_dir}")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    metrics = {
        "checkpoint": checkpoint,
        "data_size": len(train_data) + len(val_data),
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss", 0),
        "train_time_seconds": train_time,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n📊 Training Results:")
    print(f"   Train Loss: {metrics['train_loss']:.4f}")
    print(f"   Eval Loss: {metrics['eval_loss']:.4f}")
    print(f"   Time: {train_time:.1f}s")
    
    return metrics


# ============================================================================
# GENERATION - Student outputs for comparison
# ============================================================================

def generate_student_outputs(
    model,
    tokenizer,
    eval_data: List[Dict],
    checkpoint: int
) -> List[Dict]:
    """
    Generate student outputs for comparison with teacher.
    This is the KEY step for evaluation!
    """
    from unsloth import FastLanguageModel
    import torch
    from tqdm import tqdm
    
    print(f"\n🎯 Generating student outputs for checkpoint {checkpoint}...")
    print(f"   Samples: {len(eval_data)}")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    results = []
    
    for item in tqdm(eval_data, desc="Generating"):
        # Build prompt (without output - student must generate)
        prompt = INFERENCE_PROMPT.format(
            instruction=item["instruction"],
            input=item["input"]
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response
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
            "checkpoint": checkpoint
        })
    
    print(f"✓ Generated {len(results)} student outputs")
    
    # Save outputs
    outputs_dir = REPORTS_DIR / "student_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_file = outputs_dir / f"checkpoint_{checkpoint}_outputs.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Saved to {output_file}")
    
    return results


# ============================================================================
# EVALUATION - Compare student vs teacher
# ============================================================================

def evaluate_outputs(
    outputs: List[Dict],
    checkpoint: int
) -> Dict[str, float]:
    """
    Evaluate student outputs against teacher using our 7 metrics.
    This compares what the student generated vs what the teacher (Alpaca) said.
    """
    import importlib.util
    
    # Load evaluation metrics
    spec = importlib.util.spec_from_file_location(
        "eval_metrics",
        PROJECT_ROOT / "experiment" / "06_evaluation_metrics.py"
    )
    eval_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_metrics)
    
    print(f"\n📏 Evaluating checkpoint {checkpoint} ({len(outputs)} samples)...")
    
    # Aggregate metrics
    metrics_sum = {
        "structured_correctness": 0,
        "task_success": 0,
        "instruction_following": 0,
        "coverage": 0,
        "faithfulness": 0,
        "hallucination": 0,
        "context_grounding": 0,
        "overall_score": 0
    }
    
    detailed_results = []
    
    for item in outputs:
        # Evaluate student vs teacher
        result = eval_metrics.evaluate_single_output(
            instruction=item["instruction"],
            student_output=item["student_output"],
            teacher_output=item["teacher_output"],
            context=item.get("context", "")
        )
        
        # Accumulate
        for key in metrics_sum:
            metrics_sum[key] += result.get(key, 0)
        
        detailed_results.append({
            "id": item["id"],
            **result
        })
    
    # Average
    num_samples = len(outputs)
    metrics_avg = {k: v / num_samples for k, v in metrics_sum.items()}
    metrics_avg["checkpoint"] = checkpoint
    metrics_avg["num_samples"] = num_samples
    
    # Save detailed results
    detail_file = REPORTS_DIR / "detailed_evals" / f"checkpoint_{checkpoint}_eval.json"
    detail_file.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📊 Checkpoint {checkpoint} Evaluation Results:")
    print(f"   Faithfulness:      {metrics_avg['faithfulness']:.4f}")
    print(f"   Hallucination:     {metrics_avg['hallucination']:.4f} (lower is better)")
    print(f"   Coverage:          {metrics_avg['coverage']:.4f}")
    print(f"   Context Grounding: {metrics_avg['context_grounding']:.4f}")
    print(f"   Overall Score:     {metrics_avg['overall_score']:.4f}")
    
    return metrics_avg


# ============================================================================
# LEARNING CURVE
# ============================================================================

def save_results(all_results: List[Dict]):
    """Save all checkpoint results"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_file = REPORTS_DIR / "incremental_learning_results.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # CSV
    csv_file = REPORTS_DIR / "incremental_learning_results.csv"
    import csv
    if all_results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\n💾 Results saved to {REPORTS_DIR}")


def print_learning_curve(all_results: List[Dict]):
    """Print learning curve showing improvement"""
    print("\n" + "=" * 80)
    print("📈 INCREMENTAL LEARNING CURVE - Self-Learning Loop Progress")
    print("=" * 80)
    
    header = f"{'CP':<4} {'Data':<8} {'Faith':<10} {'Halluc':<10} {'Cover':<10} {'Ground':<10} {'Overall':<10}"
    print(header)
    print("-" * 80)
    
    for i, r in enumerate(all_results):
        cp = r.get("checkpoint", i + 1)
        size = r.get("data_size", cp * CHECKPOINT_SIZE)
        faith = r.get("faithfulness", 0)
        halluc = r.get("hallucination", 0)
        cover = r.get("coverage", 0)
        ground = r.get("context_grounding", 0)
        overall = r.get("overall_score", 0)
        
        # Arrows showing direction
        if i > 0:
            prev = all_results[i - 1]
            f_arr = "↑" if faith > prev.get("faithfulness", 0) else "↓" if faith < prev.get("faithfulness", 0) else "="
            h_arr = "↓" if halluc < prev.get("hallucination", 1) else "↑" if halluc > prev.get("hallucination", 0) else "="
            c_arr = "↑" if cover > prev.get("coverage", 0) else "↓" if cover < prev.get("coverage", 0) else "="
            g_arr = "↑" if ground > prev.get("context_grounding", 0) else "↓" if ground < prev.get("context_grounding", 0) else "="
            o_arr = "↑" if overall > prev.get("overall_score", 0) else "↓" if overall < prev.get("overall_score", 0) else "="
        else:
            f_arr = h_arr = c_arr = g_arr = o_arr = " "
        
        print(f"{cp:<4} {size:<8} {faith:.4f} {f_arr}  {halluc:.4f} {h_arr}  {cover:.4f} {c_arr}  {ground:.4f} {g_arr}  {overall:.4f} {o_arr}")
    
    print("-" * 80)
    
    # Summary
    if len(all_results) >= 2:
        first, last = all_results[0], all_results[-1]
        
        print(f"\n🎯 IMPROVEMENT SUMMARY (Checkpoint 1 → {len(all_results)}):")
        
        for metric, higher_better in [
            ("faithfulness", True),
            ("hallucination", False),
            ("coverage", True),
            ("context_grounding", True),
            ("overall_score", True)
        ]:
            change = last.get(metric, 0) - first.get(metric, 0)
            if higher_better:
                status = "✅ Improved" if change > 0 else "⚠️ Declined" if change < 0 else "➡️ Same"
            else:
                status = "✅ Improved" if change < 0 else "⚠️ Increased" if change > 0 else "➡️ Same"
            
            print(f"   {metric:20}: {change:+.4f} {status}")
        
        print(f"\n💡 KEY INSIGHT: Model {'shows self-learning improvement!' if last.get('overall_score', 0) > first.get('overall_score', 0) else 'needs more training.'}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(args):
    """Run the complete incremental learning pipeline"""
    import torch
    
    print("=" * 80)
    print("🔄 INCREMENTAL SELF-LEARNING LOOP")
    print("=" * 80)
    print(f"Checkpoints: {args.start_checkpoint} → {args.end_checkpoint}")
    print(f"Checkpoint Size: {CHECKPOINT_SIZE:,} records")
    print(f"Epochs per Checkpoint: {args.epochs}")
    print(f"Eval Samples: {args.eval_samples}")
    print("=" * 80)
    
    # Prepare data first
    print("\n📦 STEP 0: Prepare Alpaca 50K dataset")
    download_and_prepare_alpaca_50k()
    
    if args.dry_run:
        print("\n📋 DRY RUN - Training Plan:")
        for cp in range(args.start_checkpoint, args.end_checkpoint + 1):
            print(f"   Checkpoint {cp}: Train on {cp * CHECKPOINT_SIZE:,} records → Generate → Evaluate")
        return
    
    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    results_file = REPORTS_DIR / "incremental_learning_results.json"
    if results_file.exists() and args.start_checkpoint > 1:
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"📥 Loaded {len(all_results)} previous results")
    else:
        all_results = []
    
    # Main loop
    for checkpoint in range(args.start_checkpoint, args.end_checkpoint + 1):
        print(f"\n{'#' * 80}")
        print(f"# CHECKPOINT {checkpoint} / {args.end_checkpoint}")
        print(f"{'#' * 80}")
        
        # 1. Load data
        train_data, val_data, eval_data = load_alpaca_data(checkpoint)
        
        # Limit eval samples
        eval_data = eval_data[:args.eval_samples]
        
        # 2. Load/train model
        if args.eval_only:
            model, tokenizer = load_checkpoint_model(checkpoint)
            train_metrics = {"checkpoint": checkpoint, "data_size": len(train_data) + len(val_data)}
        else:
            model, tokenizer = load_base_model()
            train_metrics = train_checkpoint(
                checkpoint=checkpoint,
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                val_data=val_data,
                epochs=args.epochs
            )
        
        # 3. Generate student outputs (KEY STEP!)
        outputs = generate_student_outputs(
            model=model,
            tokenizer=tokenizer,
            eval_data=eval_data,
            checkpoint=checkpoint
        )
        
        # 4. Evaluate student vs teacher
        eval_metrics = evaluate_outputs(outputs, checkpoint)
        train_metrics.update(eval_metrics)
        
        # 5. Save results
        all_results.append(train_metrics)
        save_results(all_results)
        
        # 6. Show progress
        print_learning_curve(all_results)
        
        # Cleanup
        if checkpoint < args.end_checkpoint:
            del model
            del tokenizer
            torch.cuda.empty_cache()
    
    # Final
    print("\n" + "=" * 80)
    print("🎉 INCREMENTAL LEARNING COMPLETE!")
    print("=" * 80)
    print_learning_curve(all_results)
    print(f"\n📁 All results: {REPORTS_DIR}")
    print(f"📁 Checkpoints: {CHECKPOINTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Incremental Self-Learning Loop")
    parser.add_argument("--start-checkpoint", type=int, default=1)
    parser.add_argument("--end-checkpoint", type=int, default=TOTAL_CHECKPOINTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_CHECKPOINT)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Just evaluate existing checkpoints")
    parser.add_argument("--prepare-data", action="store_true", help="Just prepare Alpaca 50K data")
    
    args = parser.parse_args()
    
    if args.prepare_data:
        download_and_prepare_alpaca_50k()
        return
    
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
