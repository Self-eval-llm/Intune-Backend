"""
Generate comparison report showing improvement after prompt tuning.
Compares original metrics vs tuned metrics.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase import create_client, Client

# Import from reorganized modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client, int8_to_decimal

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def int8_to_decimal_old(value):
    """Convert INT8 metric to decimal"""
    if value is None:
        return 0.0
    return round(value / 10000, 4)


def fetch_tuned_records(supabase: Client) -> List[Dict[str, Any]]:
    """Fetch records that have been tuned (have actual_output_tuned)"""
    print("Fetching tuned records from Supabase...")
    
    response = supabase.table("inference_results")\
        .select("*")\
        .not_.is_("actual_output_tuned", "null")\
        .execute()
    
    records = response.data
    print(f"✓ Fetched {len(records)} tuned records")
    return records


def calculate_improvement(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average improvement across all metrics"""
    
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
    
    for record in records:
        for metric in metrics:
            original = int8_to_decimal(record.get(metric))
            tuned = int8_to_decimal(record.get(f"{metric}_tuned"))
            
            if original > 0:  # Avoid division by zero
                improvement_pct = ((tuned - original) / original) * 100
                improvements[metric].append({
                    'original': original,
                    'tuned': tuned,
                    'improvement_pct': improvement_pct,
                    'improvement_abs': tuned - original
                })
    
    # Calculate averages
    summary = {}
    for metric in metrics:
        if improvements[metric]:
            avg_original = sum(x['original'] for x in improvements[metric]) / len(improvements[metric])
            avg_tuned = sum(x['tuned'] for x in improvements[metric]) / len(improvements[metric])
            avg_improvement_pct = sum(x['improvement_pct'] for x in improvements[metric]) / len(improvements[metric])
            avg_improvement_abs = sum(x['improvement_abs'] for x in improvements[metric]) / len(improvements[metric])
            
            summary[metric] = {
                'avg_original': round(avg_original, 4),
                'avg_tuned': round(avg_tuned, 4),
                'avg_improvement_pct': round(avg_improvement_pct, 2),
                'avg_improvement_abs': round(avg_improvement_abs, 4),
                'count': len(improvements[metric])
            }
    
    return summary


def display_report(summary: Dict[str, Any], total_records: int):
    """Display improvement report"""
    
    print("\n" + "=" * 100)
    print("PROMPT TUNING IMPROVEMENT REPORT")
    print("=" * 100)
    print(f"\nTotal Records Analyzed: {total_records}")
    print(f"Model: Gemma 3:1b (Ollama)")
    print(f"Strategy: One-Shot Learning\n")
    
    print("=" * 100)
    print(f"{'Metric':<25} {'Original':<12} {'Tuned':<12} {'Absolute Δ':<15} {'% Change':<12}")
    print("=" * 100)
    
    # Define metric display order and names
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
            original = data['avg_original']
            tuned = data['avg_tuned']
            abs_change = data['avg_improvement_abs']
            pct_change = data['avg_improvement_pct']
            
            # Color coding for terminal (green for improvement, red for decline)
            change_symbol = "↑" if abs_change > 0 else "↓" if abs_change < 0 else "="
            
            print(f"{display_name:<25} {original:<12.4f} {tuned:<12.4f} {abs_change:>+14.4f} {pct_change:>+11.2f}%")
    
    print("=" * 100)
    
    # Summary insights
    overall_data = summary.get('overall', {})
    if overall_data:
        overall_improvement = overall_data.get('avg_improvement_pct', 0)
        
        print("\n📊 KEY INSIGHTS:")
        print("-" * 100)
        
        if overall_improvement > 0:
            print(f"✅ Overall Score improved by {overall_improvement:.2f}%")
            print(f"   From {overall_data['avg_original']:.4f} to {overall_data['avg_tuned']:.4f}")
        elif overall_improvement < 0:
            print(f"⚠️  Overall Score decreased by {abs(overall_improvement):.2f}%")
            print(f"   From {overall_data['avg_original']:.4f} to {overall_data['avg_tuned']:.4f}")
        else:
            print(f"➡️  Overall Score remained the same at {overall_data['avg_original']:.4f}")
        
        print("\n🎯 BEST IMPROVEMENTS:")
        # Find top 3 improvements
        sorted_metrics = sorted(
            [(k, v['avg_improvement_pct']) for k, v in summary.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for i, (metric, improvement) in enumerate(sorted_metrics, 1):
            print(f"   {i}. {metric_names[metric]}: +{improvement:.2f}%")
        
        print("\n⚠️  AREAS NEEDING ATTENTION:")
        # Find areas with negative improvement
        declined = [(k, v['avg_improvement_pct']) for k, v in summary.items() if v['avg_improvement_pct'] < 0]
        
        if declined:
            declined.sort(key=lambda x: x[1])
            for metric, decline in declined:
                print(f"   - {metric_names[metric]}: {decline:.2f}%")
        else:
            print("   None - All metrics improved! 🎉")
    
    print("\n" + "=" * 100)


def export_detailed_report(records: List[Dict[str, Any]], summary: Dict[str, Any]):
    """Export detailed report to JSON"""
    
    report = {
        "summary": summary,
        "total_records": len(records),
        "model": "Gemma 3:1b (Ollama)",
        "strategy": "One-Shot Learning",
        "detailed_records": []
    }
    
    # Add sample of detailed comparisons
    for record in records[:10]:  # First 10 as examples
        detail = {
            "id": record.get("id"),
            "input": record.get("input", "")[:100],
            "metrics_comparison": {}
        }
        
        for metric in ['answer_relevancy', 'faithfulness', 'overall']:
            original = int8_to_decimal(record.get(metric))
            tuned = int8_to_decimal(record.get(f"{metric}_tuned"))
            detail["metrics_comparison"][metric] = {
                "original": original,
                "tuned": tuned,
                "improvement": round(tuned - original, 4)
            }
        
        report["detailed_records"].append(detail)
    
    # Save to file
    filename = "prompt_tuning_report.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed report exported to: {filename}")


def main():
    """Main execution function"""
    print("=" * 100)
    print("GENERATING PROMPT TUNING IMPROVEMENT REPORT")
    print("=" * 100)
    
    # Create Supabase client
    supabase = get_supabase_client()
    
    # Fetch tuned records
    records = fetch_tuned_records(supabase)
    
    if not records:
        print("\n⚠️  No tuned records found!")
        print("Please run 'python prompt_tuning.py' first.")
        return
    
    # Calculate improvements
    print("\nCalculating improvements...")
    summary = calculate_improvement(records)
    
    # Display report
    display_report(summary, len(records))
    
    # Export detailed report
    print("\nExporting detailed report...")
    export_detailed_report(records, summary)
    
    print("\n✅ Report generation completed!")


if __name__ == "__main__":
    main()
