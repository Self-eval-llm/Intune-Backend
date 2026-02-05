# INTUNE: Self-Improving LLM via Incremental Knowledge Distillation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Unsloth-2025.11-orange.svg" alt="Unsloth">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-green.svg" alt="Supabase">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>An end-to-end framework for training compact LLMs through iterative knowledge distillation with automated evaluation and incremental learning.</b>
</p>

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Research Phases](#research-phases)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
  - [Phase 1: Teacher Selection](#phase-1-teacher-selection-4k-dataset)
  - [Phase 2: Incremental Learning](#phase-2-incremental-learning-50k-dataset)
- [Evaluation Framework](#evaluation-framework)
- [Experimental Results](#experimental-results)
- [Technical Implementation](#technical-implementation)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
- [Team](#team)
- [Citation](#citation)

---

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities, but their deployment remains challenging due to computational requirements. **INTUNE** addresses this through a two-phase knowledge distillation framework:

1. **Teacher Selection**: Systematic comparison of teacher models (Alpaca-7B vs GPT-OSS-20B) using 7 model-agnostic evaluation metrics on 4,000 samples
2. **Incremental Learning**: Progressive knowledge transfer from the selected teacher to a compact student model (Gemma 3:1B) through 10 stages on 50,000 samples

Our framework achieves **57.2% win rate** for the Alpaca teacher over GPT-OSS-20B, with **40.8% lower hallucination**, demonstrating that smaller, well-tuned teachers can outperform larger models for knowledge distillation tasks.

---

## Key Contributions

| Contribution | Description |
|-------------|-------------|
| **Teacher Selection Framework** | Novel methodology to systematically compare teacher models using 7 model-agnostic metrics without LLM-as-judge bias |
| **Hallucination-Aware Evaluation** | Demonstrated that hallucination rate is a critical differentiator—Alpaca achieves 0.1814 vs OSS's 0.3063 |
| **Incremental Learning Pipeline** | 10-stage progressive training (5K→50K) enabling fine-grained analysis of knowledge transfer dynamics |
| **Comprehensive Metric Suite** | 7 automated metrics covering structure, task success, instruction following, coverage, faithfulness, hallucination, and context grounding |
| **Reproducible Framework** | End-to-end pipeline with Supabase persistence, Colab compatibility, and detailed logging |

---

## Research Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTUNE RESEARCH PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 1: TEACHER SELECTION                        │   │
│  │                         (4K Dataset)                                 │   │
│  │                                                                      │   │
│  │   Stanford Alpaca ──→ Gemma 3:1B ──→ tuned_alpaca                   │   │
│  │   GPT-OSS-20B     ──→ Gemma 3:1B ──→ tuned_oss                      │   │
│  │                            ↓                                         │   │
│  │              7-Metric Evaluation Matrix                              │   │
│  │                            ↓                                         │   │
│  │              🏆 Winner: Alpaca (57.2% win rate)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   PHASE 2: INCREMENTAL LEARNING                      │   │
│  │                         (50K Dataset)                                │   │
│  │                                                                      │   │
│  │   Stage 1:  5K samples  ──→ Finetune ──→ Evaluate ──→ ckpt1        │   │
│  │   Stage 2: 10K samples  ──→ Finetune ──→ Evaluate ──→ ckpt2        │   │
│  │   Stage 3: 15K samples  ──→ Finetune ──→ Evaluate ──→ ckpt3        │   │
│  │     ...                                                              │   │
│  │   Stage 10: 50K samples ──→ Finetune ──→ Evaluate ──→ ckpt10       │   │
│  │                            ↓                                         │   │
│  │              Learning Curve Analysis                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Phase | Dataset Size | Purpose | Status |
|-------|-------------|---------|--------|
| **Phase 1** | 4,000 samples | Teacher model comparison (Alpaca vs OSS-20B) | ✅ Complete |
| **Phase 2** | 50,000 samples | 10-stage incremental knowledge distillation | 🔄 In Progress |

---

## System Architecture

### Models

| Component | Model | Parameters | Quantization |
|-----------|-------|------------|--------------|
| **Student** | Gemma 3:1B | 1 Billion | 4-bit (bnb) |
| **Teacher 1** | Stanford Alpaca | 7 Billion | 4-bit (bnb) |
| **Teacher 2** | GPT-OSS-20B | 20 Billion | 4-bit (bnb) |

### Fine-tuning Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Method** | LoRA (Low-Rank Adaptation) | Memory-efficient fine-tuning |
| **Rank (r)** | 16 | Balance between capacity and efficiency |
| **Alpha** | 16 | Standard scaling factor |
| **Dropout** | 0 | Limited data benefits from no regularization |
| **Learning Rate** | 2e-4 | Optimal for LoRA fine-tuning |
| **Batch Size** | 4 | Consumer GPU compatible (8GB VRAM) |
| **Max Seq Length** | 2048 | Full context utilization |
| **Epochs** | 3 | Prevent overfitting on limited data |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | Supabase (PostgreSQL) | Persistent storage for 50K samples + results |
| **Training** | Google Colab T4 / Local RTX 4060 | GPU compute |
| **ML Framework** | Unsloth + HuggingFace Transformers | Efficient fine-tuning |
| **Optimization** | 4-bit Quantization (bitsandbytes) | Memory reduction |

---

## Methodology

### Phase 1: Teacher Selection (4K Dataset)

#### Objective
Determine which teacher model produces better fine-tuned students by comparing two knowledge distillation approaches:
- **Approach A**: Gemma 3:1B fine-tuned on Alpaca-7B outputs
- **Approach B**: Gemma 3:1B fine-tuned on GPT-OSS-20B outputs

#### Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: TEACHER SELECTION                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Data Preparation                                            │
│  ────────────────────────                                            │
│  • Download Stanford Alpaca dataset (52K instructions)               │
│  • Sample 4,000 diverse instructions                                 │
│  • Store in Supabase table: modelComp                                │
│                                                                      │
│  Step 2: Teacher Output Generation                                   │
│  ─────────────────────────────────                                   │
│  • Generate Alpaca-7B outputs for all 4K instructions                │
│  • Generate GPT-OSS-20B outputs for all 4K instructions              │
│  • Store as 'sevenb' and 'twentyb' columns                           │
│                                                                      │
│  Step 3: Student Fine-tuning                                         │
│  ──────────────────────────                                          │
│  • Fine-tune Gemma 3:1B on Alpaca outputs → tuned_alpaca            │
│  • Fine-tune Gemma 3:1B on OSS-20B outputs → tuned_oss              │
│  • LoRA adapters saved separately                                    │
│                                                                      │
│  Step 4: Evaluation                                                  │
│  ─────────────────                                                   │
│  • Generate outputs from both tuned models                           │
│  • Compute 7 evaluation metrics                                      │
│  • Statistical comparison (chi-square test)                          │
│                                                                      │
│  Step 5: Teacher Selection                                           │
│  ────────────────────────                                            │
│  • Winner: Alpaca (57.2% win rate, p < 0.0001)                       │
│  • Selected for Phase 2 training                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### Task Categories

| Category | Count | Description | Priority Metrics |
|----------|-------|-------------|------------------|
| **general_qa** | 1,432 | Open-ended questions | All metrics equally |
| **classification_analysis** | 962 | Categorization tasks | Coverage, Instruction Following |
| **creative_generative** | 543 | Creative writing | Task Success, Instruction Following |
| **language_editing** | 434 | Translation, editing | Faithfulness, Coverage |
| **math_logic** | 338 | Mathematical reasoning | Task Success, Coverage |
| **technical_code** | 286 | Code generation | Structured Correctness, Task Success |
| **explanatory** | 1 | Explanation tasks | All metrics |

---

### Phase 2: Incremental Learning (50K Dataset)

#### Objective
Train the student model progressively on increasingly larger subsets (5K→10K→...→50K) to:
1. Measure learning curve dynamics
2. Identify optimal training data size
3. Analyze convergence behavior

#### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: INCREMENTAL LEARNING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Supabase Table: modelcomp_50k                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  id | input | context | sevenb | student_output | ckpt1..10 cols   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Stage Execution (for stage n = 1 to 10):                                   │
│  ────────────────────────────────────────                                   │
│                                                                             │
│  1. FETCH cumulative training data (n × 5,000 samples)                      │
│     └── Stage 1: 5K, Stage 2: 10K, ..., Stage 10: 50K                      │
│                                                                             │
│  2. FINETUNE Gemma 3:1B with LoRA                                           │
│     └── Training time: ~15-30 min per 5K on T4 GPU                          │
│                                                                             │
│  3. GENERATE outputs for all 50K samples                                    │
│     └── Store in: student_output_ckpt{n}                                    │
│                                                                             │
│  4. EVALUATE using 7-metric framework                                       │
│     └── Store in: score_ckpt{n}                                            │
│                                                                             │
│  5. COMPARE with previous checkpoint                                        │
│     └── Log improvement/degradation metrics                                 │
│                                                                             │
│  6. SAVE checkpoint and proceed to next stage                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Data Schema (modelcomp_50k)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT | Primary key (1-50000) |
| `checkpoint` | INT | Which 5K batch (1-10) |
| `input` | TEXT | Instruction/prompt |
| `context` | TEXT | Optional context |
| `sevenb` | TEXT | Teacher output (Alpaca-7B) |
| `student_output` | TEXT | Base student output (pre-finetuning) |
| `generation_latency` | DECIMAL | Base generation time (seconds) |
| `student_output_ckpt1` | TEXT | Output after Stage 1 training |
| `score_ckpt1` | DECIMAL | Similarity score after Stage 1 |
| `latency_ckpt1` | DECIMAL | Generation latency after Stage 1 |
| ... | ... | Columns repeat for ckpt2-10 |

---

## Evaluation Framework

### 7-Metric Evaluation Suite

All metrics are **model-agnostic** and **do not require an LLM judge**, eliminating bias from using another LLM for evaluation.

| # | Metric | Range | Goal | Description |
|---|--------|-------|------|-------------|
| 1 | **Structured Correctness** | 0-1 | ↑ Higher | Validity of structured outputs (JSON, code, lists) |
| 2 | **Task Success** | 0-1 | ↑ Higher | Heuristic-based task completion measure |
| 3 | **Instruction Following** | 0-1 | ↑ Higher | Adherence to length, format, language constraints |
| 4 | **Coverage** | 0-1 | ↑ Higher | Completeness vs reference output |
| 5 | **Faithfulness** | 0-1 | ↑ Higher | Agreement with teacher/context |
| 6 | **Hallucination** | 0-1 | ↓ Lower | Ratio of ungrounded content |
| 7 | **Context Grounding** | 0-1 | ↑ Higher | Utilization of provided context |

### Metric Definitions

#### 1. Structured Correctness
Measures validity of structured outputs:
- **Code tasks**: Python AST parsing (1.0 = valid syntax)
- **JSON tasks**: Parseability check
- **List/Table tasks**: Format validation

```python
# Scoring Logic
if task == "technical_code":
    score = 1.0 if ast.parse(output) else 0.5
elif expects_json:
    score = 1.0 if json.loads(output) else 0.3
else:
    score = 0.5 + structure_bonuses
```

#### 2. Task Success Score
Heuristic-based completion measure:
- Refusal detection ("I cannot", "As an AI")
- Minimum length check (>5 words)
- Instruction relevance (cosine similarity)
- Teacher similarity (semantic overlap)

```python
if empty or refusal: return 0.1
if too_short: return 0.2
return 0.5 + 0.3 × instruction_relevance + 0.2 × teacher_similarity
```

#### 3. Instruction Following
Rubric-based scoring (0-2 per component):

| Component | Score 2 | Score 1 | Score 0 |
|-----------|---------|---------|---------|
| **Length** | Within limit | 10-50% over | >50% over |
| **Format** | Required format present | - | Missing |
| **Refusal** | No refusal | - | Contains refusal |
| **Language** | Correct language | - | Wrong language |

#### 4. Coverage Score
Measures completeness against reference:

$$\text{Coverage} = \frac{|\text{Teacher\_terms} \cap \text{Student\_terms}|}{|\text{Teacher\_terms}|}$$

#### 5. Faithfulness Score
Weighted combination of teacher and context agreement:

$$\text{Faithfulness} = 0.6 \times \cos(\text{Student}, \text{Teacher}) + 0.4 \times \cos(\text{Student}, \text{Context})$$

#### 6. Hallucination Score
Ratio of ungrounded content (lower is better):

$$\text{Hallucination} = 1 - \frac{|\text{Context\_terms} \cap \text{Output\_terms}|}{|\text{Output\_terms}|}$$

#### 7. Context Grounding Ratio
Measures context utilization:

$$\text{Context\_Grounding} = \frac{|\text{Context\_terms} \cap \text{Output\_terms}|}{|\text{Context\_terms}|}$$

### Overall Score Computation

```python
positive = [structured_correctness, task_success, instruction_following, 
            coverage, faithfulness, context_grounding]
negative = [hallucination]

overall = mean(positive) × (1 - mean(negative))
```

---

## Experimental Results

### Phase 1: Teacher Comparison Results

#### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 3,996 |
| **Alpaca Wins** | 2,286 (57.2%) |
| **OSS-20B Wins** | 1,502 (37.6%) |
| **Ties** | 208 (5.2%) |
| **Winner** | **Alpaca** |
| **Statistical Significance** | p < 0.0001 (chi-square) |

#### Metric-by-Metric Comparison

| Metric | Alpaca | OSS-20B | Δ | Winner |
|--------|--------|---------|---|--------|
| **Structured Correctness** | 0.5182 | 0.5860 | -0.0678 | OSS |
| **Task Success** | 0.5814 | 0.6247 | -0.0433 | OSS |
| **Instruction Following** | 0.9902 | 0.9943 | -0.0041 | Tie |
| **Coverage** | 0.2172 | 0.2289 | -0.0117 | OSS |
| **Faithfulness** | 0.3603 | 0.2678 | **+0.0925** | **Alpaca** |
| **Hallucination** ↓ | **0.1814** | 0.3063 | **-0.1249** | **Alpaca** |
| **Context Grounding** | 0.8572 | 0.8525 | +0.0047 | Alpaca |
| **Overall Score** | **0.5080** | 0.4246 | **+0.0834** | **Alpaca** |

#### Key Finding: Hallucination Control

```
🏆 ALPACA DOMINATES HALLUCINATION CONTROL

   Alpaca Hallucination: 0.1814 (LOWER = BETTER)
   OSS Hallucination:    0.3063
   Advantage:            40.8% fewer hallucinations
   
   ✓ Critical for reliability and trustworthiness
   ✓ Reduces false information in generated content
   ✓ Justifies selection despite lower raw metrics
```

#### Category-wise Performance

| Category | Records | Alpaca Wins | OSS Wins | Alpaca Score | OSS Score |
|----------|---------|-------------|----------|--------------|-----------|
| **general_qa** | 1,432 | 865 | 501 | 0.5080 | 0.4246 |
| **language_editing** | 434 | 316 | 102 | 0.4583 | 0.2471 |
| **math_logic** | 338 | 215 | 104 | 0.4696 | 0.3484 |
| **classification_analysis** | 962 | 470 | 414 | 0.3990 | 0.3353 |
| **creative_generative** | 543 | 285 | 243 | 0.4879 | 0.4534 |
| **technical_code** | 286 | 134 | 138 | 0.4576 | 0.4315 |

#### Context Analysis

| Condition | Records | Alpaca Wins | OSS Wins | Alpaca Score | OSS Score |
|-----------|---------|-------------|----------|--------------|-----------|
| **With Context** | 1,729 | 1,206 (69.7%) | 321 | 0.2705 | 0.0686 |
| **Without Context** | 2,267 | 1,080 (47.6%) | 1,181 | 0.6163 | 0.6205 |

**Insight**: Alpaca significantly outperforms on context-aware tasks (69.7% win rate), critical for practical applications.

---

### Phase 2: Incremental Learning Results (Preliminary)

#### Checkpoint 1 (5K Training Data)

| Metric | Value |
|--------|-------|
| **Train Loss** | 1.0814 |
| **Eval Loss** | 0.9278 |
| **Training Time** | 670.22 seconds (~11 min) |
| **Overall Score** | 0.4883 |

#### Checkpoint 2 (10K Training Data)

| Metric | Value | Δ from Ckpt 1 |
|--------|-------|---------------|
| **Train Loss** | 1.0757 | -0.0057 |
| **Eval Loss** | 1.0042 | +0.0764 |
| **Training Time** | 3,169.85 seconds (~53 min) | - |
| **Overall Score** | 0.4904 | **+0.0021** |

#### Metric Evolution (Ckpt 1 → Ckpt 2)

| Metric | Checkpoint 1 | Checkpoint 2 | Change |
|--------|--------------|--------------|--------|
| Structured Correctness | 0.5211 | 0.5189 | -0.0022 |
| Task Success | 0.6072 | 0.6105 | +0.0033 |
| Instruction Following | 0.9907 | 0.9905 | -0.0002 |
| Coverage | 0.2922 | 0.2951 | +0.0029 |
| Faithfulness | 0.4092 | 0.4151 | +0.0059 |
| Hallucination ↓ | 0.2587 | 0.2573 | **-0.0014** |
| Context Grounding | 0.8226 | 0.8273 | +0.0047 |
| **Overall Score** | 0.4883 | **0.4904** | **+0.0021** |

---

## Technical Implementation

### Dependencies

```
# Core ML
torch>=2.0.0
transformers>=4.40.0
unsloth>=2025.11
trl>=0.7.0
bitsandbytes>=0.41.0

# Evaluation
scikit-learn>=1.3.0
rouge-score>=0.1.2
nltk>=3.8.0
sentence-transformers>=2.2.0

# Infrastructure
supabase>=2.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# Data Processing
pandas>=2.0.0
datasets>=2.14.0
```

### Hardware Requirements

| Configuration | VRAM | Speed | Recommended For |
|--------------|------|-------|-----------------|
| **RTX 4060 8GB** | 8 GB | ~12-16 sec/record | Development/Testing |
| **Colab T4** | 16 GB | ~3-5 sec/record | Production Training |
| **RTX 4090** | 24 GB | ~1-2 sec/record | Fast Iteration |

---

## Repository Structure

```
Intune_Backend/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env                                # Supabase credentials (not tracked)
│
├── experiment/                         # Core experiment scripts
│   ├── 01_data_download_alpaca.py     # Download Stanford Alpaca dataset
│   ├── 02_data_prepare_4k.py          # Prepare 4K sample dataset
│   ├── 03_gen_base_gemma.py           # Generate base Gemma outputs
│   ├── 04a_train_finetune_alpaca.py   # Finetune with Alpaca teacher
│   ├── 04b_gen_teacher_oss20b.py      # Generate OSS-20B outputs
│   ├── 05_data_label.py               # Label dataset by task type
│   ├── 06_eval_metrics.py             # 7-metric evaluation module
│   ├── 06a_gen_tuned_alpaca.py        # Generate tuned Alpaca outputs
│   ├── 07_eval_compare_teachers.py    # Compare teachers → select winner
│   ├── 08_gen_context.py              # Generate context embeddings
│   ├── 09_report_analytical.py        # Generate analytical reports
│   ├── 10_data_upload_50k.py          # Upload 50K dataset to Supabase
│   ├── 11_gen_base_student.py         # Generate base student outputs (50K)
│   ├── 12_train_incremental.py        # 10-stage incremental training
│   ├── EVALUATION_METRICS.md          # Detailed metric documentation
│   └── README.md                       # Experiment pipeline docs
│
├── colab/                              # Google Colab notebooks
│   ├── base_student_colab.ipynb       # Generate base student (50K)
│   └── finetune_incremental_colab.ipynb # Incremental finetuning
│
├── sql/                                # Database schemas
│   ├── 01_schema_setup.sql            # Initial Supabase setup
│   ├── 02_schema_eval_matrix.sql      # Evaluation columns
│   ├── 03_schema_incremental_tables.sql # Incremental learning tables
│   └── 04_schema_50k_checkpoints.sql  # 50K checkpoint columns
│
├── scripts/                            # Utility scripts
│   ├── model_convert_gguf.py          # Convert to GGUF format
│   ├── model_create_ollama.py         # Create Ollama model
│   ├── model_merge_lora.py            # Merge LoRA adapters
│   └── report_merge_results.py        # Merge result files
│
├── reports/                            # Generated reports
│   ├── teacher_comparison_report.json # Phase 1 comparison data
│   ├── teacher_comparison_analytical_report.txt # Phase 1 analysis
│   └── incremental_learning/          # Phase 2 results
│       ├── incremental_learning_results.json
│       └── detailed_evals/            # Per-checkpoint evaluations
│
├── models/                             # Saved models (git-ignored)
│   ├── gemma-alpaca-teacher/          # Alpaca-tuned LoRA
│   ├── gemma-finetuned/               # Current best model
│   └── gemma-finetuned.gguf           # GGUF export
│
├── data/                               # Data files
│   └── experiment/                     # Experiment datasets
│       ├── alpaca_50k_prepared.json   # Full 50K dataset
│       └── experiment_4k.json         # Phase 1 4K dataset
│
├── app/                                # FastAPI application
│   └── app.py                         # Inference API
│
└── config/                             # Configuration files
```

---

## Usage Guide

### Prerequisites

```bash
# Clone repository
git clone https://github.com/Self-eval-llm/Intune-Backend.git
cd Intune-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Running Phase 1 (Teacher Comparison)

```bash
# Step 1: Download Alpaca dataset
python experiment/01_data_download_alpaca.py

# Step 2: Prepare 4K sample
python experiment/02_data_prepare_4k.py

# Step 3: Generate base outputs
python experiment/03_gen_base_gemma.py

# Step 4a: Finetune with Alpaca
python experiment/04a_train_finetune_alpaca.py

# Step 4b: Generate OSS-20B outputs
python experiment/04b_gen_teacher_oss20b.py

# Step 5: Label dataset
python experiment/05_data_label.py

# Step 6: Generate tuned outputs
python experiment/06a_gen_tuned_alpaca.py

# Step 7: Compare teachers
python experiment/07_eval_compare_teachers.py
```

### Running Phase 2 (Incremental Learning)

#### Option A: Google Colab (Recommended)

1. Upload `colab/base_student_colab.ipynb` to Colab
2. Enable T4 GPU: Runtime → Change runtime type → T4 GPU
3. Add Supabase credentials in Cell 2
4. Run all cells to generate base student outputs
5. Upload `colab/finetune_incremental_colab.ipynb`
6. Set `STAGE = 1` and run all cells
7. Repeat for stages 2-10

#### Option B: Local GPU

```bash
# Generate base student outputs
python experiment/11_gen_base_student.py

# Run incremental stages
python experiment/12_train_incremental.py --stage 1
python experiment/12_train_incremental.py --stage 2
# ... continue for stages 3-10
```

### Time Estimates (Colab T4)

| Task | Time |
|------|------|
| Base Student Generation (50K) | ~50-60 hours |
| Finetune per Stage (5K) | ~15 min |
| Finetune (50K cumulative) | ~2-3 hours |
| Generate per Stage (50K) | ~4-6 hours |
| **Total 10 Stages** | **~60-80 hours** |

---

## Team

| Name | Roll Number | Role |
|------|-------------|------|
| **Radhakrishna Bharuka** | 24BDS063 | Lead Developer, ML Pipeline |
| **Abhang Pawar** | 24BDS054 | Evaluation Framework |
| **Nilesh Dwivedi** | 24BDS048 | Data Engineering |
| **Rushikesh Masalkar** | 24BDS040 | Infrastructure, Deployment |

**Institution**: Indian Institute of Technology (IIT)  
**Course**: B.Tech Data Science

---

## Citation

If you use this work, please cite:

```bibtex
@misc{intune2026,
  title={INTUNE: Self-Improving LLM via Incremental Knowledge Distillation},
  author={Bharuka, Radhakrishna and Pawar, Abhang and Dwivedi, Nilesh and Masalkar, Rushikesh},
  year={2026},
  howpublished={\url{https://github.com/Self-eval-llm/Intune-Backend}},
  note={A framework for training compact LLMs through iterative knowledge distillation}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Stanford CRFM** for the Alpaca dataset
- **Unsloth** for efficient fine-tuning infrastructure
- **Google Colab** for accessible GPU compute
- **Supabase** for database infrastructure

---

<p align="center">
  <b>Built with ❤️ for advancing efficient LLM training</b>
</p>
