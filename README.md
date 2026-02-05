# INTUNE: Self-Improving LLM Framework

An end-to-end framework for training, evaluating, and iteratively improving Large Language Models with automated feedback loops and **incremental learning**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.11-orange.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase%202-success.svg)]()

---

## 🎯 Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Teacher Comparison (Alpaca vs OSS-20B) on 4K dataset | ✅ Complete |
| **Phase 2** | 50K Incremental Learning with 10 Stages | 🔄 In Progress |

---

## Team Members

| Name                      | Roll Number |
|---------------------------|-------------|
| Radhakrishna Bharuka      | 24BDS063    |
| Abhang Pawar              | 24BDS054    |
| Nilesh Dwivedi            | 24BDS048    |
| Rushikesh Masalkar        | 24BDS040    |

---

## Table of Contents

- [Overview](#overview)
- [Phase 2: Incremental Learning](#phase-2-incremental-learning)
- [Google Colab Setup](#google-colab-setup)
- [Demo Video](#demo-video)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation Process](#installation-process)
- [Execution Guide](#execution-guide)
- [Evaluation Metrics](#evaluation-metrics)
- [API Documentation](#api-documentation)
- [Technology Stack](#technology-stack)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This framework implements a self-improving Large Language Model system that:

- Automatically collects training data from user interactions
- Evaluates model quality using 8 comprehensive metrics
- Fine-tunes models using efficient LoRA (Low-Rank Adaptation) adapters
- Measures improvements quantitatively with before/after comparisons
- **NEW: Implements 10-stage incremental learning on 50K dataset**
- Operates continuously with background workers
- Scales efficiently on consumer GPUs (8GB VRAM minimum)

---

## Phase 2: Incremental Learning

### 📊 Experiment Overview

Phase 2 implements **incremental learning** where the student model (Gemma 3:1B) learns progressively from the teacher (Alpaca-7B) through 10 stages.

```
┌─────────────────────────────────────────────────────────────────┐
│                 INCREMENTAL LEARNING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  50K Dataset (Supabase: modelcomp_50k)                         │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐ │
│  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ 5K  │ │
│  │ S1  │ S2  │ S3  │ S4  │ S5  │ S6  │ S7  │ S8  │ S9  │ S10 │ │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘ │
│                                                                 │
│  Stage 1: Train on 5K   → Generate → Eval → student_output_ckpt1│
│  Stage 2: Train on 10K  → Generate → Eval → student_output_ckpt2│
│  Stage 3: Train on 15K  → Generate → Eval → student_output_ckpt3│
│  ...                                                            │
│  Stage 10: Train on 50K → Generate → Eval → student_output_ckpt10│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📁 Supabase Table: `modelcomp_50k`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT | Primary key |
| `input` | TEXT | Instruction/prompt |
| `context` | TEXT | Optional context |
| `sevenb` | TEXT | Teacher output (Alpaca-7B) |
| `student_output` | TEXT | Base student output (before finetuning) |
| `student_output_ckpt1-10` | TEXT | Output after each stage |
| `score_ckpt1-10` | DECIMAL | Similarity score per stage |
| `latency_ckpt1-10` | DECIMAL | Generation latency per stage |

### 🚀 Phase 2 Execution Steps

```bash
# Step 1: Add checkpoint columns (run in Supabase SQL Editor)
# File: sql/04_schema_50k_checkpoints.sql

# Step 2: Generate base student outputs (before finetuning)
python experiment/11_gen_base_student.py

# Step 3: Run incremental learning stages 1-10
python experiment/12_train_incremental.py --stage 1
python experiment/12_train_incremental.py --stage 2
# ... continue for stages 3-10
```

---

## Google Colab Setup

### 🌐 Why Use Colab?

| Local RTX 4060 | Colab T4 (Free) |
|----------------|-----------------|
| ~12-16 sec/record | ~3-5 sec/record |
| 7 days for 50K | 2 days for 50K |
| Your electricity | Free GPU hours |

### 📓 Available Notebooks

| Notebook | Purpose | Location |
|----------|---------|----------|
| `base_student_colab.ipynb` | Generate base student outputs for 50K | `colab/` |
| `finetune_incremental_colab.ipynb` | Finetune + generate per stage | `colab/` |

### 🔧 Step-by-Step Colab Instructions

#### 1️⃣ Upload Notebook to Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select `colab/base_student_colab.ipynb` or `colab/finetune_incremental_colab.ipynb`

#### 2️⃣ Enable T4 GPU

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** from Hardware accelerator dropdown
3. Click **Save**

#### 3️⃣ Get Supabase Credentials

From your local `.env` file, copy:
```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_service_role_key
```

#### 4️⃣ Run Base Student Notebook

1. **Cell 1**: Install dependencies (~2 min)
2. **Cell 2**: Paste your Supabase credentials
3. **Cell 3**: Load model (~1-2 min)
4. **Cell 4**: Check remaining records
5. **Cell 5-6**: Setup and test generation
6. **Cell 7**: **Start main loop** (runs until complete or timeout)
7. **Cell 8**: Check final progress

#### 5️⃣ Run Finetuning Notebook

1. **Cell 2**: Set `STAGE = 1` and Supabase credentials
2. **Cells 3-4**: Load model with LoRA adapters
3. **Cells 5-6**: Fetch and format training data
4. **Cell 7**: Finetune (~15-30 min per stage)
5. **Cells 9-10**: Generate outputs for all 50K
6. **Cell 11**: Evaluate and compare with base
7. **Cell 12**: See summary and next steps

#### 6️⃣ Continue Next Stage

1. Change `STAGE = 2` in Cell 2
2. Click **Runtime → Restart runtime**
3. Run all cells again

### ⏱️ Time Estimates

| Task | Time on T4 |
|------|------------|
| Base Student (50K) | ~50-60 hours (4-5 sessions) |
| Finetune (5K) | ~15 min |
| Finetune (50K) | ~2-3 hours |
| Generate (50K) | ~4-6 hours |
| **Total 10 Stages** | ~60-80 hours |

### 💡 Tips for Colab Free Tier

- Free tier gives ~12 hours per session
- Sessions timeout after ~90 min of inactivity
- Progress saves to Supabase after EACH record (safe to stop)
- Can run overnight, but keep browser tab open
- If disconnected, just restart and it continues from where it left off

---

## Demo Video

**2-Minute Working Demo**

[![Demo Video Preview](docs/intune_landingpage.png)](https://github.com/Self-eval-llm/Intune-Backend/blob/main/docs/demo_video.mp4)

**[Click here to watch the full demo video](https://github.com/Self-eval-llm/Intune-Backend/blob/main/docs/demo_video.mp4)**

The demo video demonstrates:
- User interactions through the chat interface
- Real-time response generation from the model
- Background evaluation workers computing metrics
- Fine-tuning process triggering automatically
- Before and after comparison of model improvements

---

## Features

### Automated Training Pipeline
- Continuous data collection from user interactions
- Automatic dataset preparation and validation
- Threshold-based fine-tuning trigger (configurable)

### Comprehensive Evaluation System
- 8 distinct quality metrics for thorough assessment
- Before and after comparison reports
- Real-time metric computation

### Efficient Fine-tuning
- LoRA-based fine-tuning for memory efficiency
- Runs on consumer GPUs with 8GB+ VRAM
- Checkpoint saving every 100 steps

### Background Workers
- Asynchronous metric computation
- Automated fine-tuning workflow
- Continuous monitoring and processing

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   SELF-IMPROVING LLM FRAMEWORK                  │
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  │   FRONTEND   │◄───────►│   BACKEND    │◄───────►│   DATABASE   │
│  │  (React/Vue) │  REST   │   (FastAPI)  │  SQL    │  (Supabase)  │
│  │              │   API   │              │         │              │
│  │  Port: 5173  │         │  Port: 8000  │         │    Cloud     │
│  └──────────────┘         └──────────────┘         └──────────────┘
│                                   │                                    
│                                   │                                    
│                          ┌────────┴────────┐                          
│                          │                 │                          
│                    ┌─────▼─────┐    ┌─────▼─────┐                   
│                    │  Worker 1  │    │  Worker 2  │                   
│                    │ eval_first │    │eval_finetune│                  
│                    │ (Metrics)  │    │ (Training) │                   
│                    └────────────┘    └────────────┘                   
│                          │                 │                          
│                          └────────┬────────┘                          
│                                   │                                    
│                          ┌────────▼────────┐                          
│                          │  OLLAMA SERVER  │                          
│                          │  Port: 11434    │                          
│                          │                 │                          
│                          │  - Gemma 1B     │                          
│                          │  - GPT-OSS 20B  │                          
│                          │  - Fine-tuned   │                          
│                          └─────────────────┘                          
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVEMENT LOOP                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: USER INTERACTION
  ┌─────────────────┐
  │ User submits    │
  │ a question      │
  └────────┬────────┘
           │
           ▼
Step 2: RESPONSE GENERATION
  ┌─────────────────┐
  │ Gemma 1B Model  │
  │ generates answer│
  └────────┬────────┘
           │
           ▼
Step 3: DATABASE STORAGE
  ┌─────────────────────┐
  │ Save to Supabase    │
  │ status: 'created'   │
  └────────┬────────────┘
           │
           ▼
Step 4: FIRST EVALUATION
  ┌─────────────────────────────────┐
  │ Compute 8 metrics:              │
  │ - Answer Relevancy              │
  │ - Contextual Precision          │
  │ - Faithfulness                  │
  │ - Toxicity                      │
  │ - Overall Score                 │
  │ - And 3 more                    │
  │ status: 'done'                  │
  └────────┬────────────────────────┘
           │
           ▼
Step 5: DATA ACCUMULATION
  ┌─────────────────────┐
  │ Collect N records   │
  │ (default: 5000)     │
  └────────┬────────────┘
           │
           ▼
Step 6: FINE-TUNING
  ┌─────────────────────────────────┐
  │ Trigger at threshold            │
  │ - Prepare training data         │
  │ - Apply LoRA adapters           │
  │ - Train for 3 epochs            │
  │ - Save improved model           │
  └────────┬────────────────────────┘
           │
           ▼
Step 7: FINAL EVALUATION
  ┌─────────────────────────────────┐
  │ Re-evaluate with fine-tuned     │
  │ model and compare results:      │
  │                                 │
  │ Base Model → Fine-tuned Model   │
  │ Score: 0.65 → Score: 0.78       │
  │ Improvement: +20%               │
  └────────┬────────────────────────┘
           │
           ▼
Step 8: DEPLOY AND REPEAT
  ┌─────────────────────┐
  │ Use improved model  │
  │ for new interactions│
  │ Loop back to Step 1 │
  └─────────────────────┘
```

---

## Repository Structure

```
Intune_Backend/
│
├── .env                          # Environment variables (Supabase credentials)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Core dependencies
├── requirements_finetune.txt     # Fine-tuning dependencies (Unsloth)
├── README.md                     # This file
│
├── app/                          # APPLICATION LAYER
│   ├── app.py                    # Main FastAPI server (Port 8000)
│   ├── eval_first.py             # Worker 1: Base metrics evaluation
│   ├── eval_finetune.py          # Worker 2: Fine-tuning and final evaluation
│   └── README.md                 # API documentation
│
├── colab/                        # GOOGLE COLAB NOTEBOOKS
│   ├── base_student_colab.ipynb  # Generate base student outputs (50K)
│   └── finetune_incremental_colab.ipynb  # Incremental finetuning stages
│
├── experiment/                   # EXPERIMENT SCRIPTS (Numbered)
│   ├── 01_data_download_alpaca.py      # Download Alpaca dataset
│   ├── 02_data_prepare_4k.py           # Prepare 4K subset
│   ├── 03_gen_base_gemma.py            # Generate base Gemma outputs
│   ├── 04a_train_finetune_alpaca.py    # Finetune with Alpaca teacher
│   ├── 04b_gen_teacher_oss20b.py       # Generate OSS-20B outputs
│   ├── 05_data_label.py                # Label dataset
│   ├── 06_eval_metrics.py              # Compute evaluation metrics
│   ├── 06a_gen_tuned_alpaca.py         # Generate tuned model outputs
│   ├── 07_eval_compare_teachers.py     # Compare Alpaca vs OSS-20B
│   ├── 08_gen_context.py               # Generate context
│   ├── 09_report_analytical.py         # Generate analytical report
│   ├── 10_data_upload_50k.py           # Upload 50K to Supabase
│   ├── 11_gen_base_student.py          # Generate base student outputs
│   ├── 12_train_incremental.py         # Incremental learning stages
│   ├── EVALUATION_METRICS.md           # Metrics documentation
│   └── README.md                       # Experiment documentation
│
├── src/                          # SOURCE CODE LAYER
│   ├── database/                 # Database abstraction
│   │   └── supabase_client.py    # Supabase connection and utilities
│   ├── data_generation/          # Data pipeline
│   │   ├── teacher.py            # Generate training examples
│   │   ├── student.py            # Generate base outputs
│   │   └── prepare_data.py       # Format for training
│   ├── training/                 # Model fine-tuning
│   │   └── finetune.py           # LoRA-based training
│   ├── evaluation/               # Quality assessment
│   │   ├── update_metrics.py     # Compute base metrics
│   │   ├── evaluate_finetuned.py # Compare base vs tuned
│   │   ├── evaluate_finetuned_batch.py  # Batch evaluation
│   │   ├── evaluate_ollama.py    # Test deployed models
│   │   └── generate_report.py    # Create comparison reports
│   └── metrics/                  # Evaluation engine
│       └── llm_eval.py           # 8 metrics implementation
│
├── scripts/                      # UTILITY SCRIPTS
│   ├── model_convert_gguf.py     # Convert model to GGUF format
│   ├── model_create_ollama.py    # Create Ollama model
│   ├── model_merge_lora.py       # Merge LoRA adapters
│   └── report_merge_results.py   # Merge evaluation results
│
├── sql/                          # DATABASE SCHEMAS
│   ├── 01_schema_setup.sql       # Initial table setup
│   ├── 02_schema_eval_matrix.sql # Evaluation matrix columns
│   ├── 03_schema_incremental_tables.sql  # Incremental learning tables
│   └── 04_schema_50k_checkpoints.sql     # 50K checkpoint columns
│
├── models/                       # TRAINED MODELS
│   ├── gemma-finetuned.gguf      # Quantized model for Ollama
│   └── gemma-finetuned-lora/     # LoRA adapters
│
├── data/experiment/              # DATASETS
│   ├── alpaca_data_raw.json      # Raw Alpaca dataset
│   ├── alpaca_50k_prepared.json  # Prepared 50K dataset
│   └── experiment_4k.json        # 4K experiment dataset
│
├── reports/                      # EVALUATION RESULTS
│   ├── finetune_eval_results.json        # Finetuning results
│   ├── teacher_comparison_report.json    # Teacher comparison
│   └── incremental_learning/             # Incremental learning results
│
├── docs/                         # DOCUMENTATION AND MEDIA
│   ├── AI_report.pdf             # Project report
│   ├── AI_PPT.pptx               # Presentation
│   └── demo_video.mp4            # Demo video
│
└── config/                       # CONFIGURATION
    └── .env.example              # Environment template
```

### Component Responsibilities

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **app/** | API server and workers | `app.py`, `eval_first.py`, `eval_finetune.py` |
| **colab/** | Google Colab notebooks | `base_student_colab.ipynb`, `finetune_incremental_colab.ipynb` |
| **experiment/** | Numbered experiment scripts | `01-12_*.py` |
| **src/database/** | Data persistence | `supabase_client.py` |
| **src/data_generation/** | Create training data | `teacher.py`, `student.py`, `prepare_data.py` |
| **src/training/** | Fine-tune models | `finetune.py` |
| **src/evaluation/** | Assess quality | `update_metrics.py`, `evaluate_finetuned.py` |
| **src/metrics/** | Scoring engine | `llm_eval.py` |
| **scripts/** | Model utilities | `model_*.py`, `report_*.py` |
| **sql/** | Database schemas | `01-04_schema_*.sql` |
| **docs/** | Documentation and media | PDFs, images, video |

---

## Installation Process

### Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10 or higher | Core runtime |
| **NVIDIA GPU** | 8GB+ VRAM | Fine-tuning (RTX 4060 or better recommended) |
| **System RAM** | 16GB or more | Model loading |
| **Ollama** | Latest | Local LLM inference |
| **Supabase Account** | Free tier | Cloud database |
| **Git** | Latest | Clone repository |

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Self-eval-llm/Intune-Backend.git
cd Intune-Backend
```

---

### Step 2: Set Up Python Environment

**Using Virtual Environment (Recommended):**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Windows Command Prompt:
.\.venv\Scripts\activate.bat

# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements_finetune.txt
```

**Using Conda (Alternative):**

```bash
# Create conda environment
conda create -n llm-framework python=3.10

# Activate environment
conda activate llm-framework

# Install dependencies
pip install -r requirements_finetune.txt
```

---

### Step 3: Configure Environment Variables

```bash
# Copy the environment template
cp config/.env.example .env

# Edit .env file with your credentials
# On Windows: notepad .env
# On Linux/Mac: nano .env
```

**Add your Supabase credentials to `.env`:**

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-or-service-key

# Optional: Model Configuration
DEFAULT_MODEL=gemma3:1b
TEACHER_MODEL=gpt-oss:20b
```

**How to get Supabase credentials:**
1. Go to [supabase.com](https://supabase.com) and create a free account
2. Create a new project
3. Navigate to **Settings** → **API**
4. Copy the **Project URL** and **anon/public key**

---

### Step 4: Set Up Database

**Run SQL scripts in Supabase SQL Editor in the following order:**

1. Open Supabase Dashboard → SQL Editor
2. Execute `sql/01_schema_setup.sql` - Creates main table structure
3. Execute `sql/02_schema_eval_matrix.sql` - Adds evaluation matrix columns
4. Execute `sql/03_schema_incremental_tables.sql` - Creates incremental learning tables
5. Execute `sql/04_schema_50k_checkpoints.sql` - Adds checkpoint columns for 50K experiment

**Database Tables:**

| Table | Purpose |
|-------|---------|
| `intune_db` | Main table for Phase 1 (4K records) |
| `modelcomp_50k` | Phase 2 table (50K incremental learning) |

---

### Step 5: Set Up Ollama

**Install Ollama:**

```bash
# Windows (using winget):
winget install Ollama.Ollama

# Or download from https://ollama.ai

# Start Ollama service
ollama serve
```

**Pull Required Models:**

Open a new terminal window and run:

```bash
# Pull base model (Gemma 1B - approximately 1.5GB)
ollama pull gemma3:1b

# Pull teacher model (GPT-OSS 20B - approximately 20GB)
# Optional, only needed for data generation
ollama pull gpt-oss:20b

# Verify installation
ollama list
```

**Expected Output:**
```
NAME              ID              SIZE      MODIFIED
gemma3:1b         abc123def       1.5 GB    2 minutes ago
gpt-oss:20b       def456ghi       20 GB     5 minutes ago
```

---

### Step 6: Verify Installation

**Test database connection:**

```bash
python -c "from src.database.supabase_client import get_supabase_client; print('Database connected!' if get_supabase_client() else 'Connection failed')"
```

**Test Ollama connection:**

```bash
curl http://localhost:11434/api/tags
```

If successful, you should see a JSON response with a list of available models.

---

## Execution Guide

The framework requires **3 separate processes** running simultaneously. Each process should run in its own terminal window.

### Terminal 1: API Server (Main Application)

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start FastAPI server
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

**What it does:**
- Serves REST API endpoints
- Handles `/generate` requests from frontend
- Manages database operations
- Provides health check endpoint

**Console Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Status:** API is ready when you see "Application startup complete"

**API will be available at:** `http://localhost:8000`

---

### Terminal 2: First Evaluation Worker

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start first evaluation worker
python app\eval_first.py
```

**What it does:**
- Polls Supabase for records with `status_eval_first='created'`
- Computes 8 evaluation metrics for base model outputs
- Updates database with computed scores
- Marks records as `status_eval_first='done'`

**Console Output:**
```
INFO: Starting First Evaluation Worker...
INFO: Polling interval: 5 seconds
INFO: Found 3 records to evaluate
INFO: Evaluating record 123
INFO: Updated record 123 (Answer Relevancy: 0.7532)
INFO: Batch complete. Processed 3 records in 2.1s
```

**Status:** Worker is active and polling

---

### Terminal 3: Fine-tuning Worker

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start fine-tuning worker
python app\eval_finetune.py
```

**What it does:**
- Monitors record count in database
- Triggers fine-tuning when threshold reached (default: 5000 records)
- Trains LoRA adapters on collected data
- Evaluates fine-tuned model and updates `*_tuned` metrics
- Generates comparison reports

**Console Output (Initial):**
```
INFO: Starting Fine-tuning Worker...
INFO: Checking conditions every 60 seconds
INFO: Records collected: 47 / 5000 (0.94%)
INFO: Threshold not reached. Waiting...
```

**Console Output (When Triggered):**
```
INFO: Conditions met! Starting fine-tuning process...
INFO: Preparing training data...
INFO: Created train_dataset.jsonl (4500 examples)
INFO: Created val_dataset.jsonl (500 examples)
INFO: Starting fine-tuning with LoRA...
INFO: Epoch 1/3 - Loss: 0.8234
INFO: Epoch 2/3 - Loss: 0.6891
INFO: Epoch 3/3 - Loss: 0.5743
INFO: Fine-tuning completed successfully
INFO: Starting final evaluation...
INFO: Processed 100 records - Avg improvement: +12.3%
```

**Status:** Worker is monitoring; fine-tuning will trigger automatically

---

### Testing the System

**Method 1: Using curl (Command Line)**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

**Method 2: Using PowerShell**

```powershell
$body = @{
    prompt = "What is artificial intelligence?"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/generate `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

**Method 3: Using Python Script**

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Explain neural networks"}
)

print(response.json())
```

**Expected Response:**

```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "model": "gemma3:1b",
  "timestamp": "2025-11-16T14:30:45Z"
}
```

---

### Quick Start Workflow (Small Test)

For testing with 10 examples instead of 5000:

```bash
# Step 1: Generate 10 training examples
python src\data_generation\teacher.py --n 10 --mode continuous

# Step 2: Generate base model outputs
python src\data_generation\student.py

# Step 3: Compute metrics
python src\evaluation\update_metrics.py

# Step 4: Prepare training data
python src\data_generation\prepare_data.py

# Step 5: Fine-tune (edit finetune.py to use small dataset)
python src\training\finetune.py

# Step 6: Evaluate fine-tuned model
python src\evaluation\evaluate_finetuned.py
```

**Time Estimate:** Approximately 30 minutes for complete cycle with 10 examples

---

### Stopping the System

**Graceful Shutdown:**

In each terminal window, press:
```
Ctrl + C
```

Wait for "Shutting down gracefully..." message

**Force Stop (if needed):**

```bash
# Windows PowerShell
Get-Process python | Stop-Process

# Linux/Mac
pkill python
```

---

## Evaluation Metrics

The system uses 8 distinct metrics to evaluate model quality:

### Positive Metrics (Higher = Better)

**1. Answer Relevancy (0-1)**
- Measures how relevant the answer is to the question
- Formula: `cosine_similarity(question_tokens, answer_tokens)`

**2. Contextual Precision (0-1)**
- Measures how much of the answer is supported by the context
- Formula: `|answer ∩ context| / |answer|`

**3. Contextual Recall (0-1)**
- Measures how much of the context is covered in the answer
- Formula: `|answer ∩ context| / |context|`

**4. Contextual Relevancy (0-1)**
- Measures semantic similarity between context and answer
- Formula: `cosine_similarity(context_tokens, answer_tokens)`

**5. Faithfulness (0-1)**
- Measures alignment with reference answer and context
- Formula: `0.6 × cos(answer, reference) + 0.4 × cos(answer, context)`

### Negative Metrics (Lower = Better)

**6. Toxicity (0-1)**
- Measures presence of harmful or toxic language
- Formula: `toxic_words / total_words`
- Detection: Lexicon-based (offline, no API calls)

**7. Hallucination Rate (0-1)**
- Measures information not supported by context
- Formula: `1 - contextual_precision`

### Aggregate Metric

**8. Overall Score (0-1)**
- Balanced combination of all metrics
- Formula: `mean(positive_metrics) × (1 - mean(negative_metrics))`

### Example Comparison

**Before Fine-tuning (Base Model):**

| Metric | Score | Status |
|--------|-------|--------|
| Answer Relevancy | 0.6500 | Moderate |
| Contextual Precision | 0.5815 | Moderate |
| Contextual Recall | 0.6234 | Moderate |
| Contextual Relevancy | 0.5892 | Moderate |
| Faithfulness | 0.5815 | Moderate |
| Toxicity | 0.0234 | Good |
| Hallucination Rate | 0.4185 | Poor |
| **Overall Score** | **0.4721** | **Moderate** |

**After Fine-tuning:**

| Metric | Score | Improvement | Status |
|--------|-------|-------------|--------|
| Answer Relevancy | 0.7850 | +20.8% | Good |
| Contextual Precision | 0.7623 | +31.1% | Good |
| Contextual Recall | 0.7456 | +19.6% | Good |
| Contextual Relevancy | 0.7234 | +22.8% | Good |
| Faithfulness | 0.7067 | +21.5% | Good |
| Toxicity | 0.0156 | -33.3% | Excellent |
| Hallucination Rate | 0.2377 | -43.2% | Much Better |
| **Overall Score** | **0.6534** | **+38.4%** | **Good** |

---

## API Documentation

### Main Endpoints

#### `GET /`
Root endpoint providing API information

#### `GET /health`
Health check endpoint to monitor service status

#### `POST /generate`
Generate a response from the model

**Request:**
```json
{
  "prompt": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is a method of data analysis...",
  "model": "gemma3:1b"
}
```

### Interactive Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Detailed API docs:** See [app/README.md](app/README.md)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI |
| **Database** | Supabase (PostgreSQL) |
| **LLM Inference** | Ollama / Unsloth |
| **Base Model** | Gemma 3:1B |
| **Teacher Model** | Alpaca-7B (Phase 2 winner) |
| **Fine-tuning Library** | Unsloth |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) |
| **Metrics Engine** | Custom implementation |
| **Cloud Training** | Google Colab (T4 GPU) |
| **Programming Language** | Python 3.10+ |

---

## Phase 1 Results

### Teacher Comparison (Alpaca vs OSS-20B)

| Metric | Alpaca-7B | OSS-20B |
|--------|-----------|---------|
| Win Rate | **57.2%** | 42.8% |
| Avg Similarity | 0.723 | 0.689 |
| Latency | 2.1s | 8.4s |

**Winner: Alpaca-7B** - Selected as teacher for Phase 2

---

## Documentation

### Available Documentation Files

- **[Project Report (PDF)](docs/AI_report.pdf)** - Comprehensive technical documentation covering methodology, implementation, and results
- **[Project Presentation (PPTX)](docs/AI_PPT.pptx)** - Visual overview of architecture, workflow, and key features
- **[Results Analysis (PDF)](docs/result.pdf)** - Detailed evaluation results and performance metrics

### Screenshots and Diagrams

- **Landing Page:** [docs/intune_landingpage.png](docs/intune_landingpage.png)
- **Full Workflow Diagram:** [docs/Full_workflow.png](docs/Full_workflow.png)
- **Basic Workflow:** [docs/basic_workflow_figma.png](docs/basic_workflow_figma.png)
- **Database Schema:** [docs/db_schema.png](docs/db_schema.png)

---

## Troubleshooting

### Common Issues

**1. Ollama Connection Failed**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

**2. Model Not Found**
```bash
# Create model from Modelfile
ollama create gemma-finetuned -f Modelfile
```

**3. Supabase Connection Failed**
- Verify credentials in `.env` file
- Check network connectivity
- Ensure table `intune_db` exists

**4. Workers Not Processing**
- Check database has records with appropriate status flags
- Verify fine-tuned model exists at `models/gemma-finetuned-merged/`
- Check worker logs for specific errors

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Gemma** by Google for the base language model
- **Ollama** for local LLM inference infrastructure
- **Unsloth** for efficient fine-tuning framework
- **Supabase** for database and backend services

---

## Contact and Support

For questions, issues, or contributions:

- **GitHub Issues:** [Report a bug](https://github.com/Self-eval-llm/Intune-Backend/issues)
- **GitHub Discussions:** [Ask questions](https://github.com/Self-eval-llm/Intune-Backend/discussions)

---

## Project Status

**Active Development** - The project is actively maintained and continuously improving.

---

<div align="center">

**Built for the AI/ML Community**

[Star this repository](https://github.com/Self-eval-llm/Intune-Backend) if you find it useful!

</div>
