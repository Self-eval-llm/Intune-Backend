"""
Step 4c: Fine-tune Gemma 3:1B using GPT-OSS 20B outputs as teacher
==================================================================
- Fetches data from Supabase modelComp table
- Uses 'twentyb' (GPT-OSS 20B outputs) as expected_output
- Fine-tunes Gemma 3:1B with LoRA
- Saves model to models/gemma-oss20b-teacher/

Usage:
    python experiment/04c_finetune_with_oss20b.py
    python experiment/04c_finetune_with_oss20b.py --epochs 3
"""

import os
import sys
from pathlib import Path

# Windows compatibility - MUST be set before importing unsloth
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_PROC"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Training config
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gemma-oss20b-teacher"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
DEFAULT_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def fetch_training_data():
    """Fetch ALL data from Supabase for training with OSS 20B outputs"""
    print("Fetching training data from Supabase...")
    supabase = get_supabase_client()
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("modelComp")\
            .select("input, context, twentyb")\
            .not_.is_("twentyb", "null")\
            .range(offset, offset + page_size - 1)\
            .execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    print(f"✓ Fetched {len(all_records)} samples with OSS 20B outputs")
    return all_records


def format_for_training(data):
    """Format data for Gemma fine-tuning"""
    formatted = []
    
    for item in data:
        instruction = item["input"]
        context = item.get("context") or ""
        output = item["twentyb"]
        
        if context:
            input_text = f"Context: {context}\n\nQuestion: {instruction}"
        else:
            input_text = f"Question: {instruction}"
        
        formatted.append({
            "instruction": "Answer the following question accurately and concisely.",
            "input": input_text,
            "output": output
        })
    
    return formatted


def train(epochs: int = DEFAULT_EPOCHS):
    """Fine-tune Gemma with OSS 20B outputs as teacher"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    
    print("=" * 60)
    print("STEP 4c: FINE-TUNE GEMMA WITH GPT-OSS 20B TEACHER")
    print("=" * 60)
    
    # Fetch and format data
    raw_data = fetch_training_data()
    if not raw_data:
        print("✗ No training data found!")
        print("  Make sure to run 04b_generate_gpt_oss_supabase.py first")
        return
    
    formatted_data = format_for_training(raw_data)
    dataset = Dataset.from_list(formatted_data)
    
    print(f"✓ Training dataset: {len(dataset)} samples")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
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
    
    print("✓ Model loaded with LoRA adapters")
    
    # Format function
    def formatting_func(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"""<bos><start_of_turn>user
{examples["instruction"][i]}

{examples["input"][i]}<end_of_turn>
<start_of_turn>model
{examples["output"][i]}<end_of_turn>"""
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # Pre-tokenize to avoid Unsloth multiprocessing issues on Windows
    print("Tokenizing dataset (single-threaded for Windows)...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    
    dataset = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["instruction", "input", "output", "text"])
    print(f"✓ Tokenized {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=epochs,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=25,
        output_dir=str(OUTPUT_DIR),
        save_strategy="epoch",
        optim="adamw_8bit",
        dataloader_num_workers=0,
    )
    
    # Trainer with pre-tokenized data
    from transformers import Trainer, DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    trainer.train()
    
    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("→ Next: Run 05_evaluate_compare_teachers.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    train(epochs=args.epochs)
