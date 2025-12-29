"""
Step 5: Evaluate and Compare Both Fine-tuned Models
====================================================
- Loads both fine-tuned models (Alpaca teacher vs OSS 20B teacher)
- Generates outputs for ALL 4K samples
- Stores outputs in Supabase (tuned_alpaca, tuned_oss20b)
- Computes 8 evaluation metrics for each
- Generates comparison report and declares winner

Usage:
    python experiment/05_evaluate_compare_teachers.py
    python experiment/05_evaluate_compare_teachers.py --batch-size 4
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm
import numpy as np
from scipy import stats

load_dotenv()

from src.metrics.llm_eval import score_datapoint

# Model paths
ALPACA_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-alpaca-teacher"
OSS20B_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-oss20b-teacher"

# Config
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4
REPORT_DIR = PROJECT_ROOT / "reports"

# Metric columns for Supabase (8 metrics × 2 models = 16 columns)
METRIC_NAMES = [
    "answer_relevancy",
    "contextual_precision", 
    "contextual_recall",
    "contextual_relevancy",
    "faithfulness",
    "toxicity",
    "hallucination_rate",
    "overall"
]


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def fetch_all_data(supabase):
    """Fetch ALL samples from Supabase for evaluation"""
    print("Fetching all data from Supabase...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("modelComp")\
            .select("*")\
            .not_.is_("sevenb", "null")\
            .order("created_at")\
            .range(offset, offset + page_size - 1)\
            .execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    print(f"✓ Fetched {len(all_records)} samples")
    return all_records


def load_model(model_path: Path, model_name: str):
    """Load fine-tuned model"""
    from unsloth import FastLanguageModel
    
    print(f"\nLoading {model_name}...")
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None, None
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set padding for batch inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    FastLanguageModel.for_inference(model)
    
    print(f"✓ Loaded {model_name}")
    return model, tokenizer


def build_prompt(input_text: str, context: str = None) -> str:
    """Build prompt for generation"""
    if context:
        full_input = f"Context: {context}\n\nQuestion: {input_text}"
    else:
        full_input = f"Question: {input_text}"
    
    return f"""<bos><start_of_turn>user
Answer the following question accurately and concisely.

{full_input}<end_of_turn>
<start_of_turn>model
"""


def generate_batch(model, tokenizer, records: list) -> list:
    """Generate outputs for a batch of records"""
    import torch
    
    prompts = [build_prompt(r["input"], r.get("context")) for r in records]
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    responses = []
    for text in decoded:
        response = text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        responses.append(response)
    
    return responses


def compute_metrics(record: dict, output: str, reference_key: str = "sevenb") -> dict:
    """Compute 8 evaluation metrics"""
    item = {
        "input": record["input"],
        "expected_output": record.get(reference_key, ""),
        "context": [record["context"]] if record.get("context") else [],
        "actual_output": output
    }
    
    metrics = score_datapoint(item)
    
    # Return only the 8 numeric metrics
    return {k: round(metrics[k], 4) for k in METRIC_NAMES}


def to_int8(value):
    """Convert decimal to int8 for Supabase storage"""
    if value is None:
        return None
    return int(round(value * 10000))


def update_supabase_record(supabase, record_id: str, model_type: str, output: str, metrics: dict):
    """Update Supabase with output and metrics"""
    if model_type == "alpaca":
        update_data = {
            "tuned_alpaca": output,
            "alpaca_answer_relevancy": to_int8(metrics["answer_relevancy"]),
            "alpaca_contextual_precision": to_int8(metrics["contextual_precision"]),
            "alpaca_contextual_recall": to_int8(metrics["contextual_recall"]),
            "alpaca_contextual_relevancy": to_int8(metrics["contextual_relevancy"]),
            "alpaca_faithfulness": to_int8(metrics["faithfulness"]),
            "alpaca_toxicity": to_int8(metrics["toxicity"]),
            "alpaca_hallucination_rate": to_int8(metrics["hallucination_rate"]),
            "alpaca_overall": to_int8(metrics["overall"]),
        }
    else:  # oss20b
        update_data = {
            "tuned_oss20b": output,
            "oss20b_answer_relevancy": to_int8(metrics["answer_relevancy"]),
            "oss20b_contextual_precision": to_int8(metrics["contextual_precision"]),
            "oss20b_contextual_recall": to_int8(metrics["contextual_recall"]),
            "oss20b_contextual_relevancy": to_int8(metrics["contextual_relevancy"]),
            "oss20b_faithfulness": to_int8(metrics["faithfulness"]),
            "oss20b_toxicity": to_int8(metrics["toxicity"]),
            "oss20b_hallucination_rate": to_int8(metrics["hallucination_rate"]),
            "oss20b_overall": to_int8(metrics["overall"]),
        }
    
    try:
        supabase.table("modelComp").update(update_data).eq("id", record_id).execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Update error: {e}")
        return False


def evaluate_model(model, tokenizer, supabase, records: list, model_type: str, reference_key: str):
    """Evaluate a model on all records and store results"""
    print(f"\nEvaluating {model_type.upper()} model on {len(records)} samples...")
    
    all_metrics = []
    
    # Create batches
    batches = [records[i:i+BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc=f"{model_type} evaluation"):
        # Generate outputs for batch
        outputs = generate_batch(model, tokenizer, batch)
        
        # Compute metrics and update Supabase for each record
        for record, output in zip(batch, outputs):
            metrics = compute_metrics(record, output, reference_key)
            all_metrics.append(metrics)
            
            # Update Supabase
            update_supabase_record(supabase, record["id"], model_type, output, metrics)
    
    return all_metrics


def aggregate_metrics(results: list) -> dict:
    """Aggregate metrics across all samples"""
    aggregated = {}
    
    for metric in METRIC_NAMES:
        values = [r[metric] for r in results]
        aggregated[metric] = {
            "mean": round(np.mean(values), 4),
            "std": round(np.std(values), 4),
            "min": round(np.min(values), 4),
            "max": round(np.max(values), 4),
        }
    
    return aggregated


def statistical_comparison(alpaca_results: list, oss20b_results: list) -> dict:
    """Perform statistical comparison between models"""
    comparison = {}
    
    for metric in METRIC_NAMES:
        alpaca_vals = [r[metric] for r in alpaca_results]
        oss20b_vals = [r[metric] for r in oss20b_results]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(alpaca_vals, oss20b_vals)
        
        alpaca_mean = np.mean(alpaca_vals)
        oss20b_mean = np.mean(oss20b_vals)
        
        # Determine winner (higher is better, except toxicity/hallucination)
        lower_is_better = metric in ["toxicity", "hallucination_rate"]
        
        if p_value < 0.05:  # Statistically significant
            if lower_is_better:
                winner = "Alpaca" if alpaca_mean < oss20b_mean else "OSS-20B"
            else:
                winner = "Alpaca" if alpaca_mean > oss20b_mean else "OSS-20B"
        else:
            winner = "Tie"
        
        comparison[metric] = {
            "alpaca_mean": round(alpaca_mean, 4),
            "oss20b_mean": round(oss20b_mean, 4),
            "difference": round(oss20b_mean - alpaca_mean, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "winner": winner
        }
    
    return comparison


def declare_winner(comparison: dict) -> tuple:
    """Declare overall winner based on metric wins"""
    alpaca_wins = 0
    oss20b_wins = 0
    
    # Weight important metrics more
    weights = {
        "overall": 3,
        "faithfulness": 2,
        "answer_relevancy": 2,
        "contextual_precision": 1,
        "contextual_recall": 1,
        "contextual_relevancy": 1,
        "toxicity": 1,
        "hallucination_rate": 1
    }
    
    for metric, result in comparison.items():
        weight = weights.get(metric, 1)
        if result["winner"] == "Alpaca":
            alpaca_wins += weight
        elif result["winner"] == "OSS-20B":
            oss20b_wins += weight
    
    if alpaca_wins > oss20b_wins:
        return "Alpaca", alpaca_wins, oss20b_wins
    elif oss20b_wins > alpaca_wins:
        return "OSS-20B", alpaca_wins, oss20b_wins
    else:
        return "Tie", alpaca_wins, oss20b_wins


def generate_report(comparison: dict, alpaca_agg: dict, oss20b_agg: dict, 
                   winner: str, alpaca_wins: int, oss20b_wins: int,
                   total_samples: int):
    """Generate and save comprehensive report"""
    
    print("\n" + "=" * 80)
    print("TEACHER COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\n📊 Samples Evaluated: {total_samples}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-" * 80)
    print(f"{'Metric':<25} {'Alpaca':<12} {'OSS-20B':<12} {'Diff':<10} {'p-value':<12} {'Winner':<10}")
    print("-" * 80)
    
    for metric in METRIC_NAMES:
        result = comparison[metric]
        sig = "*" if result["significant"] else ""
        diff_str = f"{result['difference']:+.4f}"
        print(f"{metric:<25} {result['alpaca_mean']:<12.4f} {result['oss20b_mean']:<12.4f} {diff_str:<10} {result['p_value']:<12.6f} {result['winner']}{sig}")
    
    print("-" * 80)
    
    print(f"\n🏆 OVERALL WINNER: {winner}")
    print(f"   Alpaca weighted score: {alpaca_wins}")
    print(f"   OSS-20B weighted score: {oss20b_wins}")
    
    if winner != "Tie":
        print(f"\n✅ RECOMMENDATION: Use {winner} as teacher for full 50K training")
    else:
        print("\n⚖️ TIE: Consider other factors or run additional analysis")
    
    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "alpaca_metrics": alpaca_agg,
        "oss20b_metrics": oss20b_agg,
        "comparison": comparison,
        "winner": winner,
        "alpaca_weighted_score": alpaca_wins,
        "oss20b_weighted_score": oss20b_wins,
        "recommendation": f"Use {winner} for full 50K training" if winner != "Tie" else "Tie - additional analysis needed"
    }
    
    report_path = REPORT_DIR / "teacher_comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def main(batch_size: int = BATCH_SIZE):
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    
    print("=" * 80)
    print("STEP 5: EVALUATE & COMPARE TEACHERS (FULL 4K)")
    print("=" * 80)
    
    # Connect to Supabase
    supabase = get_supabase_client()
    
    # Fetch all data
    records = fetch_all_data(supabase)
    if not records:
        print("✗ No data found!")
        return
    
    # Load Alpaca-trained model
    alpaca_model, alpaca_tokenizer = load_model(ALPACA_MODEL_PATH, "Alpaca-Teacher Model")
    if alpaca_model is None:
        print("✗ Alpaca model not found. Run 04_finetune_with_alpaca.py first")
        return
    
    # Evaluate Alpaca model
    start_time = time.time()
    alpaca_results = evaluate_model(
        alpaca_model, alpaca_tokenizer, supabase, records,
        model_type="alpaca",
        reference_key="sevenb"  # Compare against Alpaca teacher outputs
    )
    alpaca_time = time.time() - start_time
    print(f"✓ Alpaca evaluation complete in {alpaca_time/60:.1f} minutes")
    
    # Clear GPU memory
    del alpaca_model, alpaca_tokenizer
    import torch
    torch.cuda.empty_cache()
    
    # Load OSS-20B-trained model
    oss20b_model, oss20b_tokenizer = load_model(OSS20B_MODEL_PATH, "OSS-20B-Teacher Model")
    if oss20b_model is None:
        print("✗ OSS-20B model not found. Run 04b_finetune_with_oss20b.py first")
        return
    
    # Evaluate OSS-20B model
    start_time = time.time()
    oss20b_results = evaluate_model(
        oss20b_model, oss20b_tokenizer, supabase, records,
        model_type="oss20b",
        reference_key="twentyb"  # Compare against OSS-20B teacher outputs
    )
    oss20b_time = time.time() - start_time
    print(f"✓ OSS-20B evaluation complete in {oss20b_time/60:.1f} minutes")
    
    # Aggregate metrics
    alpaca_agg = aggregate_metrics(alpaca_results)
    oss20b_agg = aggregate_metrics(oss20b_results)
    
    # Statistical comparison
    comparison = statistical_comparison(alpaca_results, oss20b_results)
    
    # Declare winner
    winner, alpaca_wins, oss20b_wins = declare_winner(comparison)
    
    # Generate report
    generate_report(
        comparison, alpaca_agg, oss20b_agg,
        winner, alpaca_wins, oss20b_wins,
        len(records)
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(batch_size=args.batch_size)
