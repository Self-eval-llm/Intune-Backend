"""
Evaluate fine-tuned model and compare with base model metrics.
Generates outputs with fine-tuned model, computes metrics, and creates comparison report.
"""

# Windows compatibility: disable torch.compile/Triton before imports
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from unsloth import FastLanguageModel

# Import from reorganized modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics.llm_eval import score_datapoint
from src.database.supabase_client import get_supabase_client, int8_to_decimal

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Fine-tuned model path (relative to project root)
FINETUNED_MODEL_PATH = os.path.join(project_root, 'models', 'gemma-finetuned-merged')


# Supabase client now imported from database module
# def get_supabase_client() -> Client:
#     """Create and return Supabase client"""
#     if not SUPABASE_URL or not SUPABASE_KEY:
#         raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
#     return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_records_with_metrics(supabase: Client, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch records that have metrics from base model"""
    print(f"Fetching top {limit} records with existing metrics...")
    
    response = supabase.table("inference_results")\
        .select("*")\
        .not_.is_("answer_relevancy", "null")\
        .order("id")\
        .limit(limit)\
        .execute()
    
    records = response.data
    print(f"✓ Fetched {len(records)} records")
    return records


def load_finetuned_model():
    """Load fine-tuned model for inference"""
    print("\n" + "=" * 80)
    print("LOADING FINE-TUNED MODEL")
    print("=" * 80)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # No quantization (model is small enough)
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print(f"✓ Loaded model from: {FINETUNED_MODEL_PATH}")
    print("✓ Inference mode enabled (full precision)")
    
    return model, tokenizer


def format_context(context: Any) -> str:
    """Format context into a string"""
    if not context:
        return ""
    
    if isinstance(context, list):
        return "\n".join(f"- {item}" for item in context if item)
    
    return str(context)


def generate_output_with_finetuned(model, tokenizer, record: Dict[str, Any]) -> str:
    """Generate output using fine-tuned model"""
    question = record.get("input", "")
    context = format_context(record.get("context"))
    
    # Create prompt in the same format as training
    instruction = "Answer the following question accurately and concisely based on the provided information."
    
    if context:
        input_text = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        input_text = f"Question: {question}"
    
    prompt = f"""<bos><start_of_turn>user
{instruction}

{input_text}<end_of_turn>
<start_of_turn>model
"""
    
    # Generate
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        use_cache=True
    )
    
    # Extract response
    response = tokenizer.batch_decode(outputs)[0]
    response = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    
    return response


def compute_metrics_for_output(record: Dict[str, Any], output: str) -> Dict[str, Any]:
    """Compute metrics for generated output"""
    item = {
        "input": record.get("input", ""),
        "expected_output": record.get("expected_output", ""),
        "context": record.get("context", []),
        "actual_output": output
    }
    
    metrics = score_datapoint(item)
    
    # Round numeric metrics to 4 decimals
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rounded_metrics[key] = round(value, 4)
        else:
            rounded_metrics[key] = value
    
    return rounded_metrics


def to_int8(value):
    """Convert decimal metric to int8 (multiply by 10000, round)"""
    if value is None:
        return None
    return int(round(value * 10000))


def save_to_supabase(supabase: Client, record_id: int, output: str, metrics: Dict[str, Any]) -> None:
    """Save fine-tuned output and metrics to Supabase *_tuned columns"""
    update_data = {
        "actual_output_tuned": output,
        "answer_relevancy_tuned": to_int8(metrics.get("answer_relevancy")),
        "contextual_precision_tuned": to_int8(metrics.get("contextual_precision")),
        "contextual_recall_tuned": to_int8(metrics.get("contextual_recall")),
        "contextual_relevancy_tuned": to_int8(metrics.get("contextual_relevancy")),
        "faithfulness_tuned": to_int8(metrics.get("faithfulness")),
        "toxicity_tuned": to_int8(metrics.get("toxicity")),
        "hallucination_rate_tuned": to_int8(metrics.get("hallucination_rate")),
        "overall_tuned": to_int8(metrics.get("overall"))
    }
    
    supabase.table("inference_results")\
        .update(update_data)\
        .eq("id", record_id)\
        .execute()


# int8_to_decimal now imported from database module
# def int8_to_decimal(value):
#     """Convert INT8 metric to decimal"""
#     if value is None:
#         return 0.0
#     return round(value / 10000, 4)


def evaluate_model_on_dataset(model, tokenizer, supabase: Client, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate fine-tuned model on dataset and collect results"""
    print("\n" + "=" * 80)
    print(f"EVALUATING FINE-TUNED MODEL ON {len(records)} RECORDS")
    print("=" * 80)
    
    results = []
    
    for i, record in enumerate(records, 1):
        record_id = record.get("id")
        question = record.get("input", "")[:50]
        
        print(f"\n[{i}/{len(records)}] Processing ID {record_id}: {question}...")
        
        try:
            # Generate output with fine-tuned model
            finetuned_output = generate_output_with_finetuned(model, tokenizer, record)
            
            # Compute metrics for fine-tuned output
            finetuned_metrics = compute_metrics_for_output(record, finetuned_output)
            
            # Save to Supabase *_tuned columns
            save_to_supabase(supabase, record_id, finetuned_output, finetuned_metrics)
            
            # Get base model metrics (convert from INT8)
            # Debug: Show raw INT8 values
            raw_overall = record.get('overall')
            if i <= 3:  # Show first 3 for debugging
                print(f"  DEBUG - Raw overall from DB (INT8): {raw_overall}")
            
            base_metrics = {
                'answer_relevancy': int8_to_decimal(record.get('answer_relevancy')),
                'contextual_precision': int8_to_decimal(record.get('contextual_precision')),
                'contextual_recall': int8_to_decimal(record.get('contextual_recall')),
                'contextual_relevancy': int8_to_decimal(record.get('contextual_relevancy')),
                'faithfulness': int8_to_decimal(record.get('faithfulness')),
                'toxicity': int8_to_decimal(record.get('toxicity')),
                'hallucination_rate': int8_to_decimal(record.get('hallucination_rate')),
                'overall': int8_to_decimal(record.get('overall'))
            }
            
            # Store result
            result = {
                'id': record_id,
                'input': record.get('input'),
                'base_output': record.get('actual_output'),
                'finetuned_output': finetuned_output,
                'base_metrics': base_metrics,
                'finetuned_metrics': finetuned_metrics
            }
            
            results.append(result)
            
            print(f"  Base Overall: {base_metrics['overall']:.4f}")
            print(f"  Finetuned Overall: {finetuned_metrics['overall']:.4f}")
            print(f"  Improvement: {(finetuned_metrics['overall'] - base_metrics['overall']):.4f}")
            print(f"  ✓ Saved to Supabase")
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    print(f"\n✓ Completed evaluation on {len(results)} records")
    print(f"✓ Saved {len(results)} fine-tuned outputs to Supabase")
    return results


def fetch_comparison_data(supabase: Client, record_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetch records with both base and tuned metrics from Supabase for accurate comparison"""
    print("\n" + "=" * 80)
    print("FETCHING STORED METRICS FROM SUPABASE FOR COMPARISON")
    print("=" * 80)
    
    response = supabase.table("inference_results")\
        .select("*")\
        .in_("id", record_ids)\
        .execute()
    
    records = response.data
    print(f"✓ Fetched {len(records)} records with both base and tuned metrics")
    
    # Convert INT8 to decimal for both base and tuned
    comparison_data = []
    for record in records:
        comparison_data.append({
            'id': record['id'],
            'input': record['input'],
            'base_metrics': {
                'answer_relevancy': int8_to_decimal(record.get('answer_relevancy')),
                'contextual_precision': int8_to_decimal(record.get('contextual_precision')),
                'contextual_recall': int8_to_decimal(record.get('contextual_recall')),
                'contextual_relevancy': int8_to_decimal(record.get('contextual_relevancy')),
                'faithfulness': int8_to_decimal(record.get('faithfulness')),
                'toxicity': int8_to_decimal(record.get('toxicity')),
                'hallucination_rate': int8_to_decimal(record.get('hallucination_rate')),
                'overall': int8_to_decimal(record.get('overall'))
            },
            'finetuned_metrics': {
                'answer_relevancy': int8_to_decimal(record.get('answer_relevancy_tuned')),
                'contextual_precision': int8_to_decimal(record.get('contextual_precision_tuned')),
                'contextual_recall': int8_to_decimal(record.get('contextual_recall_tuned')),
                'contextual_relevancy': int8_to_decimal(record.get('contextual_relevancy_tuned')),
                'faithfulness': int8_to_decimal(record.get('faithfulness_tuned')),
                'toxicity': int8_to_decimal(record.get('toxicity_tuned')),
                'hallucination_rate': int8_to_decimal(record.get('hallucination_rate_tuned')),
                'overall': int8_to_decimal(record.get('overall_tuned'))
            }
        })
    
    return comparison_data


def calculate_average_improvements(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average improvements across all metrics"""
    metrics = [
        'answer_relevancy',
        'contextual_precision',
        'contextual_recall',
        'contextual_relevancy',
        'faithfulness',
        'toxicity',
        'hallucination_rate',
        'overall'
    ]
    
    improvements = {metric: [] for metric in metrics}
    
    for result in results:
        base = result['base_metrics']
        finetuned = result['finetuned_metrics']
        
        for metric in metrics:
            base_val = base.get(metric, 0)
            finetuned_val = finetuned.get(metric, 0)
            
            if base_val > 0:
                improvement_pct = ((finetuned_val - base_val) / base_val) * 100
            else:
                improvement_pct = 0
            
            improvements[metric].append({
                'base': base_val,
                'finetuned': finetuned_val,
                'improvement_pct': improvement_pct,
                'improvement_abs': finetuned_val - base_val
            })
    
    # Calculate averages
    summary = {}
    for metric in metrics:
        if improvements[metric]:
            avg_base = sum(x['base'] for x in improvements[metric]) / len(improvements[metric])
            avg_finetuned = sum(x['finetuned'] for x in improvements[metric]) / len(improvements[metric])
            avg_improvement_pct = sum(x['improvement_pct'] for x in improvements[metric]) / len(improvements[metric])
            avg_improvement_abs = sum(x['improvement_abs'] for x in improvements[metric]) / len(improvements[metric])
            
            summary[metric] = {
                'avg_base': round(avg_base, 4),
                'avg_finetuned': round(avg_finetuned, 4),
                'avg_improvement_pct': round(avg_improvement_pct, 2),
                'avg_improvement_abs': round(avg_improvement_abs, 4)
            }
    
    return summary


def display_comparison_report(summary: Dict[str, Any], total_records: int):
    """Display comparison report"""
    print("\n" + "=" * 100)
    print("FINE-TUNED MODEL EVALUATION REPORT")
    print("=" * 100)
    print(f"\nTotal Records Evaluated: {total_records}")
    print(f"Base Model: Gemma 3 1B (original)")
    print(f"Fine-tuned Model: {FINETUNED_MODEL_PATH}\n")
    
    print("=" * 100)
    print(f"{'Metric':<25} {'Base Model':<15} {'Fine-tuned':<15} {'Absolute Δ':<15} {'% Change':<12}")
    print("=" * 100)
    
    metric_names = {
        'answer_relevancy': 'Answer Relevancy',
        'contextual_precision': 'Contextual Precision',
        'contextual_recall': 'Contextual Recall',
        'contextual_relevancy': 'Contextual Relevancy',
        'faithfulness': 'Faithfulness',
        'toxicity': 'Toxicity',
        'hallucination_rate': 'Hallucination Rate',
        'overall': 'Overall Score'
    }
    
    for metric, display_name in metric_names.items():
        if metric in summary:
            data = summary[metric]
            base = data['avg_base']
            finetuned = data['avg_finetuned']
            abs_change = data['avg_improvement_abs']
            pct_change = data['avg_improvement_pct']
            
            print(f"{display_name:<25} {base:<15.4f} {finetuned:<15.4f} {abs_change:>+14.4f} {pct_change:>+11.2f}%")
    
    print("=" * 100)
    
    # Key insights
    overall_data = summary.get('overall', {})
    if overall_data:
        overall_improvement = overall_data.get('avg_improvement_pct', 0)
        
        print("\n📊 KEY INSIGHTS:")
        print("-" * 100)
        
        if overall_improvement > 0:
            print(f"✅ Overall Score improved by {overall_improvement:.2f}%")
            print(f"   Base: {overall_data['avg_base']:.4f} → Fine-tuned: {overall_data['avg_finetuned']:.4f}")
        elif overall_improvement < 0:
            print(f"⚠️  Overall Score decreased by {abs(overall_improvement):.2f}%")
        else:
            print(f"➡️  Overall Score remained the same")
        
        print("\n🎯 BEST IMPROVEMENTS:")
        sorted_metrics = sorted(
            [(k, v['avg_improvement_pct']) for k, v in summary.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for i, (metric, improvement) in enumerate(sorted_metrics, 1):
            print(f"   {i}. {metric_names[metric]}: +{improvement:.2f}%")
        
        print("\n⚠️  AREAS WITH DECLINE:")
        declined = [(k, v['avg_improvement_pct']) for k, v in summary.items() if v['avg_improvement_pct'] < 0]
        
        if declined:
            declined.sort(key=lambda x: x[1])
            for metric, decline in declined:
                print(f"   - {metric_names[metric]}: {decline:.2f}%")
        else:
            print("   None - All metrics improved! 🎉")
    
    print("\n" + "=" * 100)


def save_detailed_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    """Save detailed comparison report to JSON"""
    report = {
        "model_comparison": {
            "base_model": "Gemma 3 1B (original)",
            "finetuned_model": FINETUNED_MODEL_PATH
        },
        "summary": summary,
        "total_records": len(results),
        "detailed_results": []
    }
    
    # Add detailed comparisons (first 10 as samples)
    for result in results[:10]:
        detail = {
            "id": result['id'],
            "input": result['input'][:100] + "..." if len(result.get('input', '')) > 100 else result.get('input', ''),
            "metrics_comparison": {}
        }
        
        # Add outputs only if available
        if 'base_output' in result and result['base_output']:
            detail["base_output"] = result['base_output'][:200] + "..." if len(result['base_output']) > 200 else result['base_output']
        if 'finetuned_output' in result and result['finetuned_output']:
            detail["finetuned_output"] = result['finetuned_output'][:200] + "..." if len(result['finetuned_output']) > 200 else result['finetuned_output']
        
        for metric in ['answer_relevancy', 'faithfulness', 'overall']:
            base_val = result['base_metrics'].get(metric)
            finetuned_val = result['finetuned_metrics'].get(metric)
            detail["metrics_comparison"][metric] = {
                "base": base_val,
                "finetuned": finetuned_val,
                "improvement": round(finetuned_val - base_val, 4) if base_val is not None and finetuned_val is not None else None
            }
        
        report["detailed_results"].append(detail)
    
    # Save to reports directory
    filename = f"evaluation_report_{len(results)}_records.json"
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed report saved to: {filepath}")


def main():
    """Main execution function"""
    print("=" * 100)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 100)
    
    # Ask for number of records
    print("\nHow many records to evaluate?")
    print("  1. Test on 100 records (recommended)")
    print("  2. Evaluate on all ~1200 records")
    choice = input("\nChoice (1 or 2): ").strip()
    
    limit = 100 if choice == "1" else 1200
    
    # Create Supabase client
    supabase = get_supabase_client()
    
    # Fetch records with existing metrics
    records = fetch_records_with_metrics(supabase, limit=limit)
    
    if not records:
        print("\n⚠️  No records with metrics found!")
        print("Please run 'python update_metrics.py' first to compute base model metrics.")
        return
    
    # Load fine-tuned model
    model, tokenizer = load_finetuned_model()
    
    # Evaluate on dataset (generates outputs and saves to Supabase)
    results = evaluate_model_on_dataset(model, tokenizer, supabase, records)
    
    if not results:
        print("\n⚠️  No results generated!")
        return
    
    # Fetch comparison data from Supabase (with proper INT8→decimal conversion)
    record_ids = [r['id'] for r in results]
    comparison_data = fetch_comparison_data(supabase, record_ids)
    
    # Calculate improvements using stored data
    print("\nCalculating improvements...")
    summary = calculate_average_improvements(comparison_data)
    
    # Display report
    display_comparison_report(summary, len(comparison_data))
    
    # Save detailed report
    save_detailed_report(comparison_data, summary)
    
    print("\n✅ Evaluation completed!")
    
    # Recommendation
    if choice == "1":
        overall_improvement = summary.get('overall', {}).get('avg_improvement_pct', 0)
        if overall_improvement > 10:
            print("\n💡 RECOMMENDATION:")
            print(f"   Model shows {overall_improvement:.2f}% improvement!")
            print("   Consider running full evaluation on all 1200 records.")
        elif overall_improvement > 0:
            print("\n💡 RECOMMENDATION:")
            print(f"   Model shows modest improvement ({overall_improvement:.2f}%).")
            print("   You may want to:")
            print("   - Train for more epochs")
            print("   - Adjust hyperparameters")
            print("   - Use more training data")
        else:
            print("\n⚠️  RECOMMENDATION:")
            print("   Model did not improve. Consider:")
            print("   - Reviewing training data quality")
            print("   - Adjusting learning rate")
            print("   - Training for more epochs")


if __name__ == "__main__":
    main()