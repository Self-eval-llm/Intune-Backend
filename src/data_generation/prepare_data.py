"""
Prepare dataset from Supabase for fine-tuning Gemma 3:1b
Exports data in format required by Unsloth/HuggingFace
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def fetch_all_records(supabase: Client, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch records from Supabase (default: top 100 for testing)"""
    print(f"Fetching top {limit} records from Supabase...")
    
    response = supabase.table("inference_results")\
        .select("id, input, expected_output, context")\
        .order("id")\
        .limit(limit)\
        .execute()
    
    records = response.data
    
    print(f"✓ Fetched {len(records)} records")
    return records


def format_context(context: Any) -> str:
    """Format context into a string"""
    if not context:
        return ""
    
    if isinstance(context, list):
        # Join list items into a readable format
        return "\n".join(f"- {item}" for item in context if item)
    
    return str(context)


def create_training_sample(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a training sample in instruction format.
    
    Format:
    {
        "instruction": "Answer the following question based on the provided context.",
        "input": "[Context]\n{context}\n\n[Question]\n{question}",
        "output": "{expected_answer}"
    }
    
    This format is compatible with Unsloth's instruction fine-tuning.
    """
    question = record.get("input", "").strip()
    expected_answer = record.get("expected_output", "").strip()
    context = format_context(record.get("context"))
    
    # Create the input with context if available
    if context:
        input_text = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        input_text = f"Question: {question}"
    
    return {
        "instruction": "Answer the following question accurately and concisely based on the provided information.",
        "input": input_text,
        "output": expected_answer
    }


def split_dataset(records: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """Split dataset into train and validation sets"""
    import random
    
    # Shuffle for random split
    shuffled = records.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    
    return train_data, val_data


def save_dataset(data: List[Dict[str, str]], filename: str):
    """Save dataset to JSONL format"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(data)} samples to {filename}")


def display_sample(sample: Dict[str, str], title: str):
    """Display a sample for verification"""
    print("\n" + "=" * 80)
    print(f"SAMPLE: {title}")
    print("=" * 80)
    print(f"\nInstruction:\n{sample['instruction']}\n")
    print(f"Input:\n{sample['input'][:200]}...\n")
    print(f"Output:\n{sample['output'][:200]}...\n")
    print("=" * 80)


def main():
    """Main execution function"""
    print("=" * 80)
    print("PREPARING FINE-TUNING DATASET FOR GEMMA 3:1B")
    print("=" * 80)
    
    # Ask user for number of records
    print("\nHow many records to use?")
    print("  1. Test with 100 records (recommended for testing)")
    print("  2. Use all records (~1200)")
    choice = input("\nChoice (1 or 2): ").strip()
    
    limit = 100 if choice == "1" else None
    
    # Create Supabase client
    supabase = get_supabase_client()
    
    # Fetch records
    if limit:
        records = fetch_all_records(supabase, limit=limit)
    else:
        # Fetch all records
        print("Fetching all records from Supabase...")
        all_records = []
        batch_size = 1000
        offset = 0
        
        while True:
            response = supabase.table("inference_results")\
                .select("id, input, expected_output, context")\
                .order("id")\
                .range(offset, offset + batch_size - 1)\
                .execute()
            
            batch = response.data
            if not batch:
                break
            
            all_records.extend(batch)
            print(f"  Fetched {len(all_records)} records so far...", end="\r")
            
            if len(batch) < batch_size:
                break
            
            offset += batch_size
        
        print(f"\n✓ Fetched {len(all_records)} total records")
        records = all_records
    
    if not records:
        print("\n⚠️  No records found!")
        return
    
    print(f"\nTotal records available: {len(records)}")
    
    # Create training samples
    print("\nFormatting samples for instruction fine-tuning...")
    training_samples = [create_training_sample(record) for record in records]
    
    # Split into train/validation (80/20)
    print("\nSplitting dataset (80% train, 20% validation)...")
    train_data, val_data = split_dataset(training_samples, train_ratio=0.8)
    
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Validation samples: {len(val_data)}")
    
    # Save datasets
    print("\nSaving datasets...")
    save_dataset(train_data, "train_dataset.jsonl")
    save_dataset(val_data, "val_dataset.jsonl")
    
    # Display sample
    display_sample(train_data[0], "Training Sample")
    
    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETED!")
    print("=" * 80)
    print("\nFiles created:")
    print("  - train_dataset.jsonl (training data)")
    print("  - val_dataset.jsonl (validation data)")
    print("\nNext step: Run 'python finetune_gemma.py' to start fine-tuning")
    print("=" * 80)


if __name__ == "__main__":
    main()
