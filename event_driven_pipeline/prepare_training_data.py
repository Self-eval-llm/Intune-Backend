"""
Create Training Data for Finetune Pipeline
==========================================

This script creates training and validation datasets from the existing intune_db data
in the format expected by the finetune.py script.
"""

import os
import sys
import json
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.database.supabase_client import get_supabase_client

def fetch_training_data():
    """Fetch data from intune_db that can be used for training"""
    try:
        supabase = get_supabase_client()

        # Fetch records that have both expected and actual outputs
        response = supabase.table("intune_db")\
            .select("input,expected_output,actual_output")\
            .not_.is_("expected_output", "null")\
            .not_.is_("actual_output", "null")\
            .execute()

        return response.data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def create_training_samples():
    """Create training samples in the required format"""

    # Sample training data based on your test questions
    training_samples = [
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "Explain photosynthesis in simple terms.",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create glucose and release oxygen."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "What is 25 * 4?",
            "output": "25 * 4 = 100"
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "Name three programming languages.",
            "output": "Three popular programming languages are Python, JavaScript, and Java."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "What causes rain?",
            "output": "Rain is caused when water vapor in clouds condenses into droplets heavy enough to fall to the ground."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "How do you make tea?",
            "output": "To make tea: boil water, add tea leaves or tea bag to cup, pour hot water over tea, steep for 3-5 minutes, then remove tea bag or strain leaves."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet in our solar system."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "Explain what HTTP stands for.",
            "output": "HTTP stands for HyperText Transfer Protocol, which is used for transferring data over the web."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "What is the difference between a list and a tuple in Python?",
            "output": "Lists are mutable (can be changed) and use square brackets [], while tuples are immutable (cannot be changed) and use parentheses ()."
        },
        {
            "instruction": "Answer the following question accurately and concisely based on the provided information.",
            "input": "How many continents are there?",
            "output": "There are 7 continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America."
        }
    ]

    # Try to fetch additional data from database
    db_data = fetch_training_data()

    for record in db_data:
        if record.get('input') and record.get('expected_output'):
            training_samples.append({
                "instruction": "Answer the following question accurately and concisely based on the provided information.",
                "input": record['input'].replace(' (Manual Test #', ' (Question #').split(' (Question #')[0],  # Clean up test suffixes
                "output": record['expected_output']
            })

    return training_samples

def save_datasets():
    """Create and save training and validation datasets"""

    print("📝 Creating training datasets...")

    # Get training samples
    samples = create_training_samples()
    print(f"   Total samples: {len(samples)}")

    # Shuffle samples
    random.shuffle(samples)

    # Split into train/validation (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"   Training samples: {len(train_samples)}")
    print(f"   Validation samples: {len(val_samples)}")

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training dataset
    train_file = output_dir / "train_dataset.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ Training data saved: {train_file}")

    # Save validation dataset
    val_file = output_dir / "val_dataset.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ Validation data saved: {val_file}")

    return train_file, val_file

def main():
    """Main function"""
    print("🚀 Creating training data for finetune pipeline...")

    train_file, val_file = save_datasets()

    print("\n📋 Next steps:")
    print("   1. Training data is ready for finetuning")
    print("   2. Run: python3 src/training/finetune.py")
    print("   3. Or let the pipeline trigger it automatically")
    print("\n✅ Training data preparation complete!")

if __name__ == "__main__":
    main()