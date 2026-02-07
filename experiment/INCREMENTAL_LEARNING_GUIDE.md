# Incremental Learning Pipeline - Complete Guide

> **File:** `experiment/12_train_incremental.py`  
> **Purpose:** Train a student model (gemma-3-1b) incrementally using teacher outputs (sevenb) across 10 checkpoints of 5000 records each.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Database Schema](#database-schema)
5. [Status Flow](#status-flow)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Metrics Explained](#metrics-explained)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## Overview

### What is Incremental Learning?

This pipeline implements **incremental learning** where a smaller student model (1B parameters) learns from a larger teacher model (7B parameters) in stages. Instead of training on all 50,000 records at once, we:

1. Divide data into **10 checkpoints** (5,000 records each)
2. Train the model on checkpoint 1, evaluate improvement
3. Continue to checkpoint 2 using the improved model from checkpoint 1
4. Repeat until all 10 checkpoints are complete

### Why Incremental?

- **Memory Efficient:** Only 5K records loaded at a time
- **Progressive Learning:** Model builds on previous knowledge
- **Checkpointing:** Can resume from any checkpoint if interrupted
- **Measurable Progress:** Track improvement at each stage

### Key Components

| Component           | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| **Student Model**   | `unsloth/gemma-3-1b-it-bnb-4bit` (1B params, 4-bit quantized) |
| **Teacher Model**   | sevenb (7B params) - outputs stored in Supabase               |
| **Training Method** | LoRA (Low-Rank Adaptation) fine-tuning                        |
| **Database**        | Supabase (PostgreSQL)                                         |
| **Evaluation**      | 11 comprehensive metrics                                      |

---

## Prerequisites

### System Requirements

| Requirement | Minimum   | Recommended |
| ----------- | --------- | ----------- |
| **GPU**     | 8GB VRAM  | 16GB+ VRAM  |
| **RAM**     | 16GB      | 32GB        |
| **Storage** | 20GB free | 50GB free   |
| **Python**  | 3.9+      | 3.10+       |
| **CUDA**    | 11.8+     | 12.0+       |

### Software Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth
pip install trl transformers datasets
pip install supabase python-dotenv
pip install rouge-score nltk
pip install tqdm
```

### Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-key
```

### Pre-populated Data Required

Before running the pipeline, your Supabase table must have:

| Column           | Required Data                                  |
| ---------------- | ---------------------------------------------- |
| `input`          | Instruction/prompt text                        |
| `context`        | Optional context for the instruction           |
| `sevenb`         | Teacher model output (ground truth)            |
| `student_output` | Base student model output (before fine-tuning) |
| `checkpoint`     | Checkpoint number (1-10)                       |

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INCREMENTAL LEARNING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         ┌──────────┐     │
│  │Checkpoint│ → │Checkpoint│ → │Checkpoint│ → ... → │Checkpoint│     │
│  │    1     │    │    2     │    │    3     │         │    10    │     │
│  │ (5000)   │    │ (5000)   │    │ (5000)   │         │ (5000)   │     │
│  └──────────┘    └──────────┘    └──────────┘         └──────────┘     │
│       │               │               │                     │           │
│       ▼               ▼               ▼                     ▼           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         ┌──────────┐     │
│  │  Model   │ → │  Model   │ → │  Model   │ → ... → │  Model   │     │
│  │  ckpt1   │    │  ckpt2   │    │  ckpt3   │         │  ckpt10  │     │
│  └──────────┘    └──────────┘    └──────────┘         └──────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Single Checkpoint Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT N WORKFLOW (5 Steps)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌────────────┐   ┌────┐│
│  │  SCORE  │ → │ FINETUNE │ → │OUTPUT_TUNED │ → │SCORE_TUNED │ → │DONE││
│  └─────────┘   └──────────┘   └─────────────┘   └────────────┘   └────┘│
│       │             │               │                 │             │   │
│       ▼             ▼               ▼                 ▼             ▼   │
│  Score base    Train model    Generate new      Score tuned    Calculate│
│  student vs    on teacher     outputs with      outputs vs     improve- │
│  teacher       outputs        finetuned model   teacher        ment     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SUPABASE TABLE: modelcomp_50k                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │  INPUT DATA (Pre-existing)          GENERATED DATA (Pipeline)   │   │
│   │  ─────────────────────────          ──────────────────────────  │   │
│   │  • input                            • score (base metrics)      │   │
│   │  • context                          • score_tuned (tuned)       │   │
│   │  • sevenb (teacher)                 • student_output_tuned      │   │
│   │  • student_output (base)            • latency_tuned             │   │
│   │  • checkpoint (1-10)                • improvement               │   │
│   │  • generation_latency               • status                    │   │
│   │                                     • 11 metric columns         │   │
│   │                                     • 11 metric_tuned columns   │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Existing Columns (DO NOT MODIFY)

| Column               | Type     | Description               |
| -------------------- | -------- | ------------------------- |
| `id`                 | uuid/int | Primary key               |
| `input`              | text     | Instruction/prompt        |
| `context`            | text     | Optional context          |
| `sevenb`             | text     | Teacher model output      |
| `student_output`     | text     | Base student output       |
| `generation_latency` | float    | Base generation time (ms) |
| `checkpoint`         | int      | Checkpoint number (1-10)  |

### New Columns (Added by SQL Migration)

#### Status Column

| Column   | Type        | Description                                                            |
| -------- | ----------- | ---------------------------------------------------------------------- |
| `status` | varchar(20) | Workflow status: score, finetune, output_tuned, score_tuned, completed |

#### Tuned Output Columns

| Column                 | Type  | Description                    |
| ---------------------- | ----- | ------------------------------ |
| `student_output_tuned` | text  | Finetuned model output         |
| `latency_tuned`        | float | Finetuned generation time (ms) |

#### Base Metrics (Before Finetuning)

| Column                   | Type  | Range | Description                                  |
| ------------------------ | ----- | ----- | -------------------------------------------- |
| `score`                  | float | 0-1   | Overall weighted score                       |
| `structured_correctness` | float | 0-1   | Format/structure accuracy                    |
| `task_success`           | float | 0-1   | Task completion rate                         |
| `instruction_following`  | float | 0-1   | Adherence to instructions                    |
| `coverage`               | float | 0-1   | Information completeness                     |
| `faithfulness`           | float | 0-1   | Factual accuracy                             |
| `hallucination`          | float | 0-1   | 1 = no hallucination, 0 = high hallucination |
| `context_grounding`      | float | 0-1   | Use of provided context                      |
| `conciseness`            | float | 0-1   | Appropriate length                           |
| `rouge1`                 | float | 0-1   | ROUGE-1 F1 score                             |
| `rougeL`                 | float | 0-1   | ROUGE-L F1 score                             |
| `bleu`                   | float | 0-1   | BLEU score                                   |

#### Tuned Metrics (After Finetuning)

Same as above with `_tuned` suffix:

- `score_tuned`, `structured_correctness_tuned`, `task_success_tuned`, etc.

#### Improvement Column

| Column        | Type  | Description                           |
| ------------- | ----- | ------------------------------------- |
| `improvement` | float | score_tuned - score (can be negative) |

---

## Status Flow

### Status Values

```
┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐     ┌───────────┐
│  score   │ ──▶ │ finetune │ ──▶ │ output_tuned│ ──▶ │ score_tuned│ ──▶ │ completed │
└──────────┘     └──────────┘     └─────────────┘     └────────────┘     └───────────┘
     │                │                  │                   │                  │
     ▼                ▼                  ▼                   ▼                  ▼
  Score base      Train on          Generate           Score tuned        Calculate
  student vs      teacher           tuned              output vs          improvement
  teacher         outputs           outputs            teacher            & report
```

### Status Transition Table

| Current Status | Step to Run           | Next Status    | What Happens                       |
| -------------- | --------------------- | -------------- | ---------------------------------- |
| `NULL`         | `--init`              | `score`        | Initialize records                 |
| `score`        | `--step score`        | `finetune`     | Calculate 11 base metrics          |
| `finetune`     | `--step finetune`     | `output_tuned` | Train LoRA adapter                 |
| `output_tuned` | `--step output_tuned` | `score_tuned`  | Generate new outputs               |
| `score_tuned`  | `--step score_tuned`  | `completed`    | Calculate 11 tuned metrics         |
| `completed`    | `--step completed`    | `completed`    | Calculate improvement, save report |

### Key Principle: Status-Based Filtering

**Each step ONLY processes records with that specific status.**

Example:

- `--step score` fetches: `WHERE checkpoint=N AND status='score'`
- Records with `status='completed'` are **never re-processed**
- This prevents duplicate work and ensures data integrity

---

## Installation & Setup

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd Intune-Backend
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements_finetune.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Step 5: Run SQL Migration

Run the SQL query provided in the [SQL Migration](#sql-migration) section to add required columns.

### Step 6: Verify Data

Ensure your Supabase table has:

- 50,000 records with `checkpoint` values 1-10
- Each checkpoint should have ~5,000 records
- Columns `input`, `sevenb`, `student_output` populated

---

## Usage Guide

### Quick Start

```bash
# 1. Check current status of checkpoint 1
python experiment/12_train_incremental.py --checkpoint 1 --step status

# 2. Initialize checkpoint (set status to 'score')
python experiment/12_train_incremental.py --checkpoint 1 --init

# 3. Run all steps automatically
python experiment/12_train_incremental.py --checkpoint 1 --run-all
```

### Command Reference

| Command          | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `--checkpoint N` | **Required.** Checkpoint number (1-10)                                           |
| `--step STATUS`  | Run specific step: score, finetune, output_tuned, score_tuned, completed, status |
| `--run-all`      | Run all 5 steps sequentially                                                     |
| `--init`         | Initialize records (set status='score')                                          |

### Running Individual Steps

```bash
# Step 1: Score base student output (requires 5000 records with status='score')
python experiment/12_train_incremental.py --checkpoint 1 --step score

# Step 2: Finetune model (requires 5000 records with status='finetune')
python experiment/12_train_incremental.py --checkpoint 1 --step finetune

# Step 3: Generate tuned outputs (requires 5000 records with status='output_tuned')
python experiment/12_train_incremental.py --checkpoint 1 --step output_tuned

# Step 4: Score tuned outputs (requires 5000 records with status='score_tuned')
python experiment/12_train_incremental.py --checkpoint 1 --step score_tuned

# Step 5: Calculate improvement (requires 5000 records with status='completed')
python experiment/12_train_incremental.py --checkpoint 1 --step completed
```

### Full Pipeline (All 10 Checkpoints)

```bash
# Run checkpoints 1-10 in sequence
for i in {1..10}; do
    echo "=== Processing Checkpoint $i ==="
    python experiment/12_train_incremental.py --checkpoint $i --init
    python experiment/12_train_incremental.py --checkpoint $i --run-all
done
```

### Checking Progress

```bash
# View detailed status with progress bars
python experiment/12_train_incremental.py --checkpoint 1 --step status
```

**Sample Output:**

```
============================================================
📊 STATUS SUMMARY - Checkpoint 1
============================================================

   Total records: 5000
   Required per step: 5000

   Status Distribution:
   ──────────────────────────────────────────────────
   ❌ null/empty       :     0 / 5000 [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%
   ✅ score            :  5000 / 5000 [██████████████████████████████] 100%
   ❌ finetune         :     0 / 5000 [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%
   ...

   ──────────────────────────────────────────────────
   📈 READINESS SUMMARY
   ──────────────────────────────────────────────────
   Next step:         --step score
   Records ready:     5,000
   Records remaining: 0
   Progress:          100.0%

   ✅ READY TO RUN: python experiment/12_train_incremental.py --checkpoint 1 --step score
============================================================
```

---

## Metrics Explained

### Semantic Metrics (from 06_eval_metrics.py)

| Metric                     | Weight | Description                          | Good Score |
| -------------------------- | ------ | ------------------------------------ | ---------- |
| **structured_correctness** | 15%    | Does output follow expected format?  | > 0.8      |
| **task_success**           | 20%    | Was the task completed successfully? | > 0.7      |
| **instruction_following**  | 15%    | Did it follow the instruction?       | > 0.8      |
| **coverage**               | 15%    | Is all required info included?       | > 0.7      |
| **faithfulness**           | 15%    | Is the information accurate?         | > 0.8      |
| **hallucination**          | 10%    | Is there made-up info? (1=good)      | > 0.8      |
| **context_grounding**      | 5%     | Does it use provided context?        | > 0.7      |
| **conciseness**            | 5%     | Is it appropriately concise?         | > 0.6      |

### Traditional NLP Metrics

| Metric      | Description                | Good Score |
| ----------- | -------------------------- | ---------- |
| **ROUGE-1** | Unigram overlap            | > 0.4      |
| **ROUGE-L** | Longest common subsequence | > 0.35     |
| **BLEU**    | N-gram precision           | > 0.2      |

### Overall Score Calculation

```python
overall_score = (
    structured_correctness * 0.15 +
    task_success * 0.20 +
    instruction_following * 0.15 +
    coverage * 0.15 +
    faithfulness * 0.15 +
    hallucination * 0.10 +
    context_grounding * 0.05 +
    conciseness * 0.05
)
```

### Improvement Calculation

```python
improvement = score_tuned - score
# Positive = model improved
# Negative = model regressed
# Zero = no change
```

---

## Configuration

### Model Configuration

```python
# In 12_train_incremental.py

MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"  # Base model
MAX_SEQ_LENGTH = 2048                          # Max context length
MAX_NEW_TOKENS = 512                           # Max generation length
RECORDS_PER_CHECKPOINT = 5000                  # Records per checkpoint
MIN_RECORDS_PER_CHECKPOINT = 5000              # Minimum required
```

### LoRA Configuration

```python
LORA_R = 16          # LoRA rank (higher = more capacity, more memory)
LORA_ALPHA = 16      # LoRA alpha (scaling factor)
LORA_DROPOUT = 0     # Dropout (0 for inference stability)

# Target modules for Gemma
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]
```

### Training Configuration

```python
TrainingArguments(
    per_device_train_batch_size=4,    # Batch size per GPU
    gradient_accumulation_steps=4,     # Effective batch = 4*4 = 16
    warmup_steps=50,                   # LR warmup
    num_train_epochs=1,                # Epochs per checkpoint
    learning_rate=2e-4,                # Peak learning rate
    fp16=True,                         # Mixed precision (or bf16)
    logging_steps=25,                  # Log every N steps
    optim="adamw_8bit",                # 8-bit Adam optimizer
    weight_decay=0.01,                 # L2 regularization
    lr_scheduler_type="linear",        # LR decay
)
```

### Adjusting for Your Hardware

**Low VRAM (8GB):**

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
MAX_SEQ_LENGTH = 1024
```

**High VRAM (24GB+):**

```python
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
MAX_SEQ_LENGTH = 4096
```

---

## Troubleshooting

### Common Issues

#### 1. "VALIDATION FAILED: Need X more records"

**Cause:** Not enough records with the required status.

**Solution:**

```bash
# Check current status
python experiment/12_train_incremental.py --checkpoint 1 --step status

# If records are uninitialized (NULL status)
python experiment/12_train_incremental.py --checkpoint 1 --init
```

#### 2. "Model not found: models/gemma-ckptN-lora"

**Cause:** Previous checkpoint's finetuning wasn't completed.

**Solution:**

```bash
# Run finetune step for the checkpoint
python experiment/12_train_incremental.py --checkpoint N --step finetune
```

#### 3. CUDA Out of Memory

**Solutions:**

1. Reduce batch size in `TrainingArguments`
2. Reduce `MAX_SEQ_LENGTH`
3. Use `gradient_checkpointing=True`
4. Close other GPU applications

#### 4. "Missing teacher output (sevenb)"

**Cause:** Some records don't have teacher outputs.

**Solution:**

1. Run teacher generation script first
2. Or manually populate the `sevenb` column

#### 5. Supabase Connection Failed

**Check:**

1. `.env` file exists with correct credentials
2. `SUPABASE_URL` is correct (no trailing slash)
3. `SUPABASE_KEY` has necessary permissions

### Debug Mode

Add verbose logging:

```python
# At top of script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reset a Checkpoint

To re-run a checkpoint from scratch:

```sql
-- Reset all records for checkpoint 1 to NULL status
UPDATE modelcomp_50k
SET status = NULL,
    score = NULL,
    score_tuned = NULL,
    student_output_tuned = NULL,
    improvement = NULL
WHERE checkpoint = 1;
```

---

## FAQ

### Q: Can I run multiple checkpoints in parallel?

**A:** No. Each checkpoint depends on the previous checkpoint's model. Run sequentially.

### Q: What if I only have 3000 records for a checkpoint?

**A:** The pipeline will not proceed. You need exactly 5000 records per checkpoint. Either:

- Collect more data
- Modify `MIN_RECORDS_PER_CHECKPOINT` (not recommended)

### Q: Can I skip a checkpoint?

**A:** Not recommended. The model builds incrementally. Skipping checkpoint 2 means checkpoint 3 uses checkpoint 1's model, losing potential learning.

### Q: How long does each checkpoint take?

**Rough estimates (RTX 3090):**
| Step | Time |
|------|------|
| score | 30-60 min |
| finetune | 2-4 hours |
| output_tuned | 1-2 hours |
| score_tuned | 30-60 min |
| completed | 5-10 min |

### Q: Can I resume if the script crashes?

**A:** Yes! Status-based workflow means:

- Completed records keep their status
- Just re-run the same command
- It will only process remaining records

### Q: How do I know if finetuning is working?

**A:** Check the improvement metric:

- Positive improvement = learning from teacher
- After 3-5 checkpoints, expect 5-15% improvement
- Diminishing returns are normal

---

## Output Files

### Model Checkpoints

```
models/
├── gemma-ckpt1-lora/    # After checkpoint 1
├── gemma-ckpt2-lora/    # After checkpoint 2
├── ...
└── gemma-ckpt10-lora/   # Final model
```

### Reports

```
reports/incremental/
├── checkpoint_1_report.json
├── checkpoint_2_report.json
├── ...
└── checkpoint_10_report.json
```

### Report Format

```json
{
  "checkpoint": 1,
  "timestamp": "2026-02-07T10:30:00",
  "records": 5000,
  "score_before": 0.4523,
  "score_after": 0.5127,
  "improvement": 0.0604,
  "improvement_pct": 13.35
}
```

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in the terminal output
3. Check Supabase data integrity
4. Open an issue in the repository

---

_Last updated: February 2026_
