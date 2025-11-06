"""
Script to reorganize project files into proper structure.
Run this after reviewing the changes.
"""

import shutil
import os

ROOT = r"c:\Users\Radhakrishna\Downloads\llm"

# Define file mappings: (source, destination)
FILE_MOVES = [
    # Data generation
    ("teacher.py", "src/data_generation/teacher.py"),
    ("student.py", "src/data_generation/student.py"),
    ("prepare_finetune_data.py", "src/data_generation/prepare_data.py"),
    
    # Training
    ("finetune_gemma.py", "src/training/finetune.py"),
    
    # Evaluation
    ("evaluate_finetuned.py", "src/evaluation/evaluate_finetuned.py"),
    ("evaluate_ollama.py", "src/evaluation/evaluate_ollama.py"),
    ("update_metrics.py", "src/evaluation/update_metrics.py"),
    ("generate_report.py", "src/evaluation/generate_report.py"),
    
    # SQL files
    ("supabase_setup.sql", "sql/supabase_setup.sql"),
    ("supabase_add_metrics.sql", "sql/supabase_add_metrics.sql"),
    ("add_tuned_columns.sql", "sql/add_tuned_columns.sql"),
    ("create_decimal_view.sql", "sql/create_decimal_view.sql"),
    
    # Utility scripts
    ("convert_to_gguf.py", "scripts/convert_to_gguf.py"),
    ("create_ollama_model.py", "scripts/create_ollama_model.py"),
    ("cleanup.ps1", "scripts/cleanup.ps1"),
    
    # Data files
    ("training_dataset.json", "data/raw/training_dataset.json"),
    ("train_dataset.jsonl", "data/processed/train_dataset.jsonl"),
    ("val_dataset.jsonl", "data/processed/val_dataset.jsonl"),
    ("output1.json", "data/raw/output1.json"),
    
    # Reports
    ("evaluation_report_100_records.json", "reports/evaluation_report_100_records.json"),
    
    # Model files (just document the location)
    # These stay in root due to size and direct references
    # ("gemma-finetuned.gguf", "models/gemma-finetuned.gguf"),
]

def reorganize():
    """Copy files to new structure (keeps originals for safety)"""
    print("=" * 80)
    print("REORGANIZING PROJECT FILES")
    print("=" * 80)
    
    for src, dest in FILE_MOVES:
        src_path = os.path.join(ROOT, src)
        dest_path = os.path.join(ROOT, dest)
        
        if not os.path.exists(src_path):
            print(f"⚠️  SKIP: {src} (not found)")
            continue
        
        # Create destination directory if needed
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        try:
            # Copy file (don't delete original yet for safety)
            shutil.copy2(src_path, dest_path)
            print(f"✓ Copied: {src} → {dest}")
        except Exception as e:
            print(f"✗ ERROR: {src} → {dest}: {e}")
    
    print("\n" + "=" * 80)
    print("✓ REORGANIZATION COMPLETE")
    print("=" * 80)
    print("\nIMPORTANT: Review the new structure before deleting original files.")
    print("Original files are preserved. Delete manually after verification.")

if __name__ == "__main__":
    reorganize()
