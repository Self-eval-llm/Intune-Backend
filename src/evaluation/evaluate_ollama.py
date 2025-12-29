"""
Evaluate fine-tuned Ollama model and compare with base model metrics.
Uses Ollama API to generate outputs, then computes metrics.
"""

import os
import json
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Import from reorganized modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics.llm_eval import score_datapoint
from src.database.supabase_client import get_supabase_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma-finetuned"

# Batch size
BATCH_SIZE = 10


def get_supabase_client() -> Client:
    """Create and return Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_records(limit: int = None) -> List[Dict[str, Any]]:
    """Fetch records from Supabase for evaluation"""
    supabase = get_supabase_client()
    
    print(f"\nFetching records from Supabase...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        query = supabase.table('inference_results').select('*').order('id').range(offset, offset + page_size - 1)
        
        if limit and offset >= limit:
            break
        
        response = query.execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched {len(all_records)} records so far...")
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    if limit:
        all_records = all_records[:limit]
    
    print(f"✓ Total fetched: {len(all_records)} records")
    return all_records


def format_prompt(input_text: str, context: Any) -> str:
    """Format prompt for Ollama (matches training format)"""
    # Parse context if it's a JSON string
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except:
            pass
    
    # Extract context text
    if isinstance(context, dict):
        context_text = context.get('text', '') or context.get('context', '')
    elif isinstance(context, list):
        context_text = '\n'.join([
            item.get('text', '') or item.get('context', '') 
            for item in context if isinstance(item, dict)
        ])
    else:
        context_text = str(context) if context else ''
    
    # Format prompt (matches training format)
    instruction = "Answer the following question accurately and concisely based on the provided information."
    
    if context_text:
        prompt = f"{instruction}\n\nContext:\n{context_text}\n\nQuestion: {input_text}"
    else:
        prompt = f"{instruction}\n\nQuestion: {input_text}"
    
    return prompt


def generate_output_with_ollama(prompt: str) -> str:
    """Generate output using Ollama API"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }, timeout=60)
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"⚠️  Ollama API error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"⚠️  Error generating output: {e}")
        return ""


def compute_metrics(record: Dict[str, Any], generated_output: str) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # Parse context
    context = record.get('context', {})
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except:
            pass
    
    # Extract context text
    if isinstance(context, dict):
        context_text = context.get('text', '') or context.get('context', '')
    elif isinstance(context, list):
        context_text = '\n'.join([
            item.get('text', '') or item.get('context', '') 
            for item in context if isinstance(item, dict)
        ])
    else:
        context_text = str(context) if context else ''
    
    # Prepare datapoint for evaluation
    datapoint = {
        'input': record['input'],
        'expected_output': record['expected_output'],
        'actual_output': generated_output,
        'context': context_text
    }
    
    # Compute metrics
    metrics = score_datapoint(datapoint)
    
    return metrics


def save_to_supabase(record_id: int, output: str, metrics: Dict[str, float]):
    """Save generated output and metrics to Supabase *_tuned columns"""
    supabase = get_supabase_client()
    
    # Prepare update data - save as regular floats
    update_data = {
        'actual_output_tuned': output,
        'answer_relevancy_tuned': metrics.get('answer_relevancy', 0),
        'contextual_precision_tuned': metrics.get('contextual_precision', 0),
        'contextual_recall_tuned': metrics.get('contextual_recall', 0),
        'contextual_relevancy_tuned': metrics.get('contextual_relevancy', 0),
        'faithfulness_tuned': metrics.get('faithfulness', 0),
        'toxicity_tuned': metrics.get('toxicity', 0),
        'hallucination_rate_tuned': metrics.get('hallucination_rate', 0),
        'overall_tuned': metrics.get('overall', 0),
        'updated_at': datetime.now().isoformat()
    }
    
    # Update record
    supabase.table('inference_results').update(update_data).eq('id', record_id).execute()


def process_records(records: List[Dict[str, Any]]):
    """Process records in batches"""
    total = len(records)
    
    print(f"\n{'=' * 100}")
    print(f"EVALUATING {total} RECORDS WITH OLLAMA MODEL")
    print(f"{'=' * 100}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}\n")
    
    results = []
    
    for i, record in enumerate(records, 1):
        print(f"\rProcessing {i}/{total}...", end='', flush=True)
        
        # Format prompt
        prompt = format_prompt(record['input'], record.get('context'))
        
        # Generate output with Ollama
        generated_output = generate_output_with_ollama(prompt)
        
        if not generated_output:
            print(f"\n⚠️  Skipping record {record['id']} - no output generated")
            continue
        
        # Compute metrics
        metrics = compute_metrics(record, generated_output)
        
        # Save to Supabase
        save_to_supabase(record['id'], generated_output, metrics)
        
        # Store result for reporting
        results.append({
            'id': record['id'],
            'input': record['input'],
            'generated_output': generated_output,
            'metrics': metrics
        })
        
        # Batch progress
        if i % BATCH_SIZE == 0:
            print(f"\n✓ Completed batch {i//BATCH_SIZE} ({i}/{total})")
    
    print(f"\n\n✓ All records processed!")
    return results


def calculate_average_improvements(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average metric improvements"""
    supabase = get_supabase_client()
    
    # Fetch updated records with both base and tuned metrics
    record_ids = [r['id'] for r in records]
    
    response = supabase.table('inference_results').select('*').in_('id', record_ids).execute()
    data = response.data
    
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
    
    improvements = {}
    
    for metric in metrics:
        base_values = []
        tuned_values = []
        
        for record in data:
            base_val = record.get(metric)
            tuned_val = record.get(f"{metric}_tuned")
            
            if base_val is not None and tuned_val is not None:
                # Base metrics are INT8, need conversion; tuned are already floats
                base_values.append(base_val / 10000.0)
                tuned_values.append(tuned_val)
        
        if base_values and tuned_values:
            avg_base = sum(base_values) / len(base_values)
            avg_tuned = sum(tuned_values) / len(tuned_values)
            absolute_diff = avg_tuned - avg_base
            percent_change = (absolute_diff / avg_base * 100) if avg_base > 0 else 0
            
            improvements[metric] = {
                'base': avg_base,
                'tuned': avg_tuned,
                'absolute_diff': absolute_diff,
                'percent_change': percent_change
            }
    
    return improvements


def print_comparison_report(improvements: Dict[str, Dict], total_records: int):
    """Print formatted comparison report"""
    print("\n" + "=" * 100)
    print("FINE-TUNED OLLAMA MODEL EVALUATION REPORT")
    print("=" * 100)
    print(f"\nTotal Records Evaluated: {total_records}")
    print(f"Base Model: Gemma 3 1B (original)")
    print(f"Fine-tuned Model: {MODEL_NAME} (Ollama)\n")
    
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
    
    for metric, stats in improvements.items():
        name = metric_names.get(metric, metric)
        base = stats['base']
        tuned = stats['tuned']
        diff = stats['absolute_diff']
        pct = stats['percent_change']
        
        # Color coding
        indicator = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        
        print(f"{indicator} {name:<23} {base:<15.4f} {tuned:<15.4f} {diff:+.4f}         {pct:+.2f}%")
    
    print("=" * 100)


def main():
    """Main evaluation pipeline"""
    print("=" * 100)
    print("OLLAMA MODEL EVALUATION")
    print("=" * 100)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    
    # Check Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ Ollama is not running. Please start Ollama first.")
            return
    except:
        print("\n❌ Cannot connect to Ollama. Please ensure Ollama is running.")
        return
    
    # Evaluate all records
    print("\n📊 Evaluating all records...")
    limit = None
    
    # Fetch records
    records = fetch_records(limit)
    
    if not records:
        print("No records found!")
        return
    
    # Process records
    results = process_records(records)
    
    # Calculate improvements
    print("\n\nCalculating metric improvements...")
    improvements = calculate_average_improvements(records)
    
    # Print comparison report
    print_comparison_report(improvements, len(records))
    
    # Save detailed report to reports directory
    report_filename = f"ollama_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, report_filename)
    
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'total_records': len(records),
            'improvements': improvements,
            'detailed_results': results[:10]  # Save first 10 for reference
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved: {report_file}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
