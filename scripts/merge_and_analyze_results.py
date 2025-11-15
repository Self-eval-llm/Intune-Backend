"""
Merge all inference results CSV files and perform comprehensive analysis
comparing metrics before and after fine-tuning.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_and_merge_csv_files(results_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from Results directory and merge them.
    
    Args:
        results_dir: Path to Results directory
        
    Returns:
        Merged DataFrame
    """
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("inference_results_rows*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Read and concatenate all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on 'id' column
    merged_df = merged_df.drop_duplicates(subset=['id'], keep='first')
    
    print(f"\nTotal records after merging: {len(merged_df)}")
    print(f"Unique records: {merged_df['id'].nunique()}")
    
    return merged_df


def calculate_percentage_change(before: float, after: float) -> float:
    """
    Calculate percentage change from before to after.
    
    Args:
        before: Value before fine-tuning
        after: Value after fine-tuning
        
    Returns:
        Percentage change
    """
    if pd.isna(before) or pd.isna(after) or before == 0:
        return np.nan
    return ((after - before) / before) * 100


def analyze_metrics(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive analysis of metrics before and after fine-tuning.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Dictionary containing analysis results
    """
    # Define metric columns (before and after fine-tuning)
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
    
    analysis = {
        'summary_statistics': {},
        'percentage_changes': {},
        'improvement_breakdown': {},
        'detailed_comparisons': {},
        'record_level_analysis': []
    }
    
    # 1. Summary Statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for metric in metrics:
        before_col = metric
        after_col = f"{metric}_tuned"
        
        if before_col in df.columns and after_col in df.columns:
            # Calculate statistics
            before_mean = df[before_col].mean()
            after_mean = df[after_col].mean()
            before_median = df[before_col].median()
            after_median = df[after_col].median()
            before_std = df[before_col].std()
            after_std = df[after_col].std()
            
            pct_change_mean = calculate_percentage_change(before_mean, after_mean)
            pct_change_median = calculate_percentage_change(before_median, after_median)
            
            analysis['summary_statistics'][metric] = {
                'before_mean': before_mean,
                'after_mean': after_mean,
                'before_median': before_median,
                'after_median': after_median,
                'before_std': before_std,
                'after_std': after_std,
                'percentage_change_mean': pct_change_mean,
                'percentage_change_median': pct_change_median
            }
            
            print(f"\n{metric.upper().replace('_', ' ')}")
            print(f"  Before: Mean={before_mean:.2f}, Median={before_median:.2f}, Std={before_std:.2f}")
            print(f"  After:  Mean={after_mean:.2f}, Median={after_median:.2f}, Std={after_std:.2f}")
            print(f"  Change: {pct_change_mean:+.2f}% (mean), {pct_change_median:+.2f}% (median)")
    
    # 2. Improvement Breakdown
    print("\n" + "="*80)
    print("IMPROVEMENT BREAKDOWN")
    print("="*80)
    
    for metric in metrics:
        before_col = metric
        after_col = f"{metric}_tuned"
        
        if before_col in df.columns and after_col in df.columns:
            # Calculate differences
            df[f'{metric}_diff'] = df[after_col] - df[before_col]
            df[f'{metric}_pct_change'] = df.apply(
                lambda row: calculate_percentage_change(row[before_col], row[after_col]), 
                axis=1
            )
            
            # Count improvements, degradations, and no change
            improved = (df[f'{metric}_diff'] > 0).sum()
            degraded = (df[f'{metric}_diff'] < 0).sum()
            unchanged = (df[f'{metric}_diff'] == 0).sum()
            total = len(df)
            
            analysis['improvement_breakdown'][metric] = {
                'improved_count': int(improved),
                'degraded_count': int(degraded),
                'unchanged_count': int(unchanged),
                'improved_percentage': (improved / total) * 100,
                'degraded_percentage': (degraded / total) * 100,
                'unchanged_percentage': (unchanged / total) * 100
            }
            
            print(f"\n{metric.upper().replace('_', ' ')}")
            print(f"  Improved: {improved} ({(improved/total)*100:.1f}%)")
            print(f"  Degraded: {degraded} ({(degraded/total)*100:.1f}%)")
            print(f"  Unchanged: {unchanged} ({(unchanged/total)*100:.1f}%)")
    
    # 3. Detailed Comparisons by Performance Bands
    print("\n" + "="*80)
    print("PERFORMANCE BAND ANALYSIS")
    print("="*80)
    
    for metric in metrics:
        before_col = metric
        after_col = f"{metric}_tuned"
        
        if before_col in df.columns and after_col in df.columns:
            # Define performance bands
            df[f'{metric}_before_band'] = pd.cut(
                df[before_col], 
                bins=[0, 3000, 5000, 7000, 10000],
                labels=['Poor (0-3000)', 'Fair (3000-5000)', 'Good (5000-7000)', 'Excellent (7000-10000)']
            )
            
            # Calculate improvement by band
            band_analysis = df.groupby(f'{metric}_before_band').agg({
                before_col: ['mean', 'count'],
                after_col: 'mean',
                f'{metric}_diff': 'mean',
                f'{metric}_pct_change': 'mean'
            }).round(2)
            
            analysis['detailed_comparisons'][metric] = band_analysis.to_dict()
            
            print(f"\n{metric.upper().replace('_', ' ')} by Performance Band:")
            print(band_analysis)
    
    # 4. Record-Level Analysis (Top Improvements and Degradations)
    print("\n" + "="*80)
    print("TOP IMPROVEMENTS AND DEGRADATIONS")
    print("="*80)
    
    for metric in metrics:
        before_col = metric
        after_col = f"{metric}_tuned"
        
        if before_col in df.columns and after_col in df.columns:
            # Top 5 improvements
            top_improvements = df.nlargest(5, f'{metric}_diff')[
                ['id', before_col, after_col, f'{metric}_diff', f'{metric}_pct_change']
            ]
            
            # Top 5 degradations
            top_degradations = df.nsmallest(5, f'{metric}_diff')[
                ['id', before_col, after_col, f'{metric}_diff', f'{metric}_pct_change']
            ]
            
            print(f"\n{metric.upper().replace('_', ' ')} - Top 5 Improvements:")
            print(top_improvements.to_string(index=False))
            
            print(f"\n{metric.upper().replace('_', ' ')} - Top 5 Degradations:")
            print(top_degradations.to_string(index=False))
    
    # 5. Correlation Analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlations between metrics
    metric_cols_before = [m for m in metrics if m in df.columns]
    metric_cols_after = [f"{m}_tuned" for m in metrics if f"{m}_tuned" in df.columns]
    
    if metric_cols_before:
        corr_before = df[metric_cols_before].corr()
        print("\nCorrelation Matrix (Before Fine-tuning):")
        print(corr_before.round(2))
    
    if metric_cols_after:
        corr_after = df[metric_cols_after].corr()
        print("\nCorrelation Matrix (After Fine-tuning):")
        print(corr_after.round(2))
    
    return analysis, df


def generate_insights(analysis: Dict, df: pd.DataFrame) -> List[str]:
    """
    Generate actionable insights from the analysis.
    
    Args:
        analysis: Analysis results
        df: DataFrame with analysis
        
    Returns:
        List of insights
    """
    insights = []
    
    insights.append("="*80)
    insights.append("KEY INSIGHTS AND RECOMMENDATIONS")
    insights.append("="*80)
    
    # Overall performance
    overall_before = analysis['summary_statistics']['overall']['before_mean']
    overall_after = analysis['summary_statistics']['overall']['after_mean']
    overall_change = analysis['summary_statistics']['overall']['percentage_change_mean']
    
    insights.append(f"\n1. OVERALL PERFORMANCE:")
    insights.append(f"   - Before fine-tuning: {overall_before:.2f}")
    insights.append(f"   - After fine-tuning: {overall_after:.2f}")
    insights.append(f"   - Change: {overall_change:+.2f}%")
    
    if overall_change > 0:
        insights.append(f"   ✓ Model shows overall improvement after fine-tuning")
    else:
        insights.append(f"   ✗ Model shows overall degradation after fine-tuning")
    
    # Best improvements
    insights.append(f"\n2. AREAS OF STRONGEST IMPROVEMENT:")
    best_metrics = sorted(
        analysis['summary_statistics'].items(),
        key=lambda x: x[1]['percentage_change_mean'],
        reverse=True
    )[:3]
    
    for i, (metric, stats) in enumerate(best_metrics, 1):
        insights.append(f"   {i}. {metric.replace('_', ' ').title()}: {stats['percentage_change_mean']:+.2f}%")
    
    # Areas needing attention
    insights.append(f"\n3. AREAS NEEDING ATTENTION:")
    worst_metrics = sorted(
        analysis['summary_statistics'].items(),
        key=lambda x: x[1]['percentage_change_mean']
    )[:3]
    
    for i, (metric, stats) in enumerate(worst_metrics, 1):
        insights.append(f"   {i}. {metric.replace('_', ' ').title()}: {stats['percentage_change_mean']:+.2f}%")
    
    # Consistency analysis
    insights.append(f"\n4. CONSISTENCY ANALYSIS:")
    for metric, breakdown in analysis['improvement_breakdown'].items():
        if breakdown['improved_percentage'] > 70:
            insights.append(f"   ✓ {metric.replace('_', ' ').title()}: Consistently improved ({breakdown['improved_percentage']:.1f}% of cases)")
        elif breakdown['degraded_percentage'] > 70:
            insights.append(f"   ✗ {metric.replace('_', ' ').title()}: Consistently degraded ({breakdown['degraded_percentage']:.1f}% of cases)")
        else:
            insights.append(f"   ≈ {metric.replace('_', ' ').title()}: Mixed results (improved: {breakdown['improved_percentage']:.1f}%, degraded: {breakdown['degraded_percentage']:.1f}%)")
    
    # Recommendations
    insights.append(f"\n5. RECOMMENDATIONS:")
    
    # Check toxicity
    if 'toxicity' in analysis['summary_statistics']:
        tox_change = analysis['summary_statistics']['toxicity']['percentage_change_mean']
        if tox_change > 10:
            insights.append(f"   ⚠ Toxicity increased by {tox_change:.1f}% - Review training data for toxic patterns")
    
    # Check hallucination
    if 'hallucination_rate' in analysis['summary_statistics']:
        hall_change = analysis['summary_statistics']['hallucination_rate']['percentage_change_mean']
        if hall_change > 10:
            insights.append(f"   ⚠ Hallucination rate increased by {hall_change:.1f}% - Consider additional grounding techniques")
    
    # Check faithfulness
    if 'faithfulness' in analysis['summary_statistics']:
        faith_change = analysis['summary_statistics']['faithfulness']['percentage_change_mean']
        if faith_change < -10:
            insights.append(f"   ⚠ Faithfulness decreased by {abs(faith_change):.1f}% - Review context alignment in training data")
    
    return insights


def save_results(analysis: Dict, df: pd.DataFrame, insights: List[str], output_dir: str):
    """
    Save analysis results to files.
    
    Args:
        analysis: Analysis dictionary
        df: DataFrame with merged data
        insights: List of insights
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save merged DataFrame
    merged_file = output_path / "merged_inference_results.csv"
    df.to_csv(merged_file, index=False)
    print(f"\n✓ Merged data saved to: {merged_file}")
    
    # Save analysis as JSON
    analysis_file = output_path / "analysis_results.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✓ Analysis results saved to: {analysis_file}")
    
    # Save insights as text
    insights_file = output_path / "insights_and_recommendations.txt"
    with open(insights_file, 'w') as f:
        f.write('\n'.join(insights))
    print(f"✓ Insights saved to: {insights_file}")
    
    # Save summary statistics as CSV
    summary_df = pd.DataFrame(analysis['summary_statistics']).T
    summary_file = output_path / "summary_statistics.csv"
    summary_df.to_csv(summary_file)
    print(f"✓ Summary statistics saved to: {summary_file}")
    
    # Save improvement breakdown as CSV
    improvement_df = pd.DataFrame(analysis['improvement_breakdown']).T
    improvement_file = output_path / "improvement_breakdown.csv"
    improvement_df.to_csv(improvement_file)
    print(f"✓ Improvement breakdown saved to: {improvement_file}")


def main():
    """
    Main function to orchestrate the analysis.
    """
    print("="*80)
    print("INFERENCE RESULTS ANALYSIS - BEFORE vs AFTER FINE-TUNING")
    print("="*80)
    
    # Define paths
    results_dir = r"c:\Users\Radhakrishna\Downloads\llm\Results"
    output_dir = r"c:\Users\Radhakrishna\Downloads\llm\reports"
    
    # Step 1: Load and merge CSV files
    print("\n[Step 1] Loading and merging CSV files...")
    merged_df = load_and_merge_csv_files(results_dir)
    
    # Step 2: Perform comprehensive analysis
    print("\n[Step 2] Performing comprehensive analysis...")
    analysis, analyzed_df = analyze_metrics(merged_df)
    
    # Step 3: Generate insights
    print("\n[Step 3] Generating insights and recommendations...")
    insights = generate_insights(analysis, analyzed_df)
    
    # Print insights
    print("\n")
    for insight in insights:
        print(insight)
    
    # Step 4: Save results
    print("\n[Step 4] Saving results...")
    save_results(analysis, analyzed_df, insights, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nTotal records analyzed: {len(merged_df)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
