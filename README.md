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
  - [Phase 1: Teacher Comparison](#phase-1-teacher-comparison-results)
  - [Phase 2: Incremental Learning (Preliminary)](#phase-2-incremental-learning-results-preliminary)
  - [Phase 2: Proposed Scientific Analysis](#phase-2-proposed-scientific-analysis-planned)
- [Technical Implementation](#technical-implementation)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
- [Team](#team)
- [Citation](#citation)

---

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities, but their deployment remains challenging due to computational requirements. **INTUNE** addresses this through a two-phase knowledge distillation framework:

1. **Teacher Selection**: Systematic comparison of teacher models (Alpaca-7B vs GPT-OSS-20B) using 7 model-agnostic evaluation metrics on 4,000 samples
2. **Incremental Learning**: Progressive knowledge transfer from the selected teacher to a compact student model (Gemma 3:1B) through 10 checkpoints on 50,000 samples, with controlled batch comparison

Our framework employs a **status-based pipeline** (score → finetune → output_tuned → score_tuned → completed) for fault-tolerant, resumable training on Google Colab T4 GPUs. Phase 1 achieves **57.2% win rate** for the Alpaca teacher over GPT-OSS-20B, with **40.8% lower hallucination**, demonstrating that smaller, well-tuned teachers can outperform larger models for knowledge distillation tasks. Phase 2 evaluates both incremental and batch training strategies using 7 core metrics plus ROUGE-1, ROUGE-L, and BLEU.

---

## Key Contributions

| Contribution | Description |
|-------------|-------------|
| **Teacher Selection Framework** | Novel methodology to systematically compare teacher models using 7 model-agnostic metrics without LLM-as-judge bias |
| **Hallucination-Aware Evaluation** | Demonstrated that hallucination rate is a critical differentiator—Alpaca achieves 0.1814 vs OSS's 0.3063 |
| **Incremental Learning Pipeline** | 10-checkpoint progressive training (5K→50K) with status-based workflow enabling fault-tolerant, resumable execution |
| **Incremental vs. Batch Comparison** | Controlled comparison of incremental (10-checkpoint) vs. batch (single-pass) training strategies |
| **Comprehensive Metric Suite** | 7 automated core metrics + ROUGE-1, ROUGE-L, BLEU covering structure, task success, instruction following, coverage, faithfulness, hallucination, and context grounding |
| **Reproducible Framework** | End-to-end pipeline with Supabase persistence, Google Colab T4 notebooks, status-based progress tracking, and detailed logging |

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
│  │   Ckpt 1:  5K samples  ──→ Finetune ──→ Score Tuned ──→ Complete   │   │
│  │   Ckpt 2: 10K samples  ──→ Finetune ──→ Score Tuned ──→ Complete   │   │
│  │   Ckpt 3: 15K samples  ──→ Finetune ──→ Score Tuned ──→ Complete   │   │
│  │     ...                                                              │   │
│  │   Ckpt 10: 50K samples ──→ Finetune ──→ Score Tuned ──→ Complete   │   │
│  │                            ↓                                         │   │
│  │   Status: score → finetune → output_tuned → score_tuned → completed │   │
│  │                            ↓                                         │   │
│  │              Learning Curve + Batch Comparison                       │   │
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
1. Measure learning curve dynamics across 10 checkpoints
2. Identify optimal training data size and diminishing returns thresholds
3. Analyze convergence behavior and catastrophic forgetting
4. Compare incremental vs. batch training strategies
5. Generate statistically significant results with confidence intervals

#### Status-Based Workflow

Phase 2 employs a **checkpoint/status-based pipeline** where each record in Supabase progresses through a well-defined state machine. This ensures fault tolerance, resumability, and fine-grained progress tracking.

```
┌──────────────────────────────────────────────────────────────────────────┐
│              STATUS-BASED RECORD LIFECYCLE (Per Checkpoint)              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐ │
│   │  score   │ ──→ │ finetune │ ──→ │ output_tuned │ ──→ │score_tuned │ │
│   └─────────┘     └──────────┘     └──────────────┘     └────────────┘ │
│       │                                                       │          │
│       │           Base scoring                   Tuned scoring │          │
│       │           complete                       complete      │          │
│       │                                                       ↓          │
│       │                                              ┌────────────┐     │
│       └─────────────────────────────────────────────→│ completed  │     │
│                                                      └────────────┘     │
│                                                                          │
│   Status Flow: score → finetune → output_tuned → score_tuned →          │
│                completed                                                 │
│                                                                          │
│   • score:        Record scored with base student (pre-finetune)         │
│   • finetune:     Ready for finetuning batch                             │
│   • output_tuned: Finetuned model output generated                       │
│   • score_tuned:  Tuned output evaluated with 7+3 metrics                │
│   • completed:    All processing done for this checkpoint                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: INCREMENTAL LEARNING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Supabase Table: modelcomp_50k                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  id | input | context | sevenb | student_output | status |         │   │
│  │  student_output_tuned | score_tuned | latency_tuned | improvement  │   │
│  │  student_output_batch | score_batch | latency_batch                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Checkpoint Execution (for checkpoint n = 1 to 10):                         │
│  ──────────────────────────────────────────────                             │
│                                                                             │
│  1. FETCH records WHERE status = 'finetune' AND checkpoint = n              │
│     └── 5,000 records per checkpoint (cumulative training)                  │
│                                                                             │
│  2. FINETUNE Gemma 3:1B with LoRA on cumulative data                        │
│     └── Ckpt 1: 5K, Ckpt 2: 10K, ..., Ckpt 10: 50K                        │
│     └── Save LoRA adapter to Google Drive / local                           │
│                                                                             │
│  3. GENERATE tuned outputs for checkpoint records                           │
│     └── Store in: student_output_tuned                                      │
│     └── Update status: 'finetune' → 'output_tuned'                         │
│                                                                             │
│  4. EVALUATE using 7-metric + ROUGE/BLEU framework                          │
│     └── Store in: score_tuned, latency_tuned                               │
│     └── Compute improvement = score_tuned - score (base)                    │
│     └── Update status: 'output_tuned' → 'score_tuned' → 'completed'        │
│                                                                             │
│  5. REPORT per-checkpoint metrics and cumulative analysis                    │
│     └── Log improvement/degradation vs previous checkpoint                  │
│     └── Save to reports/incremental_learning/                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Incremental vs. Batch Comparison

A key research contribution is the **controlled comparison** between incremental and batch training:

| Aspect | Incremental Training | Batch Training |
|--------|---------------------|----------------|
| **Approach** | Progressive: 5K → 10K → ... → 50K | Single pass: All 50K at once |
| **Checkpoints** | 10 (one per 5K increment) | 1 (final model only) |
| **Output Column** | `student_output_tuned` | `student_output_batch` |
| **Score Column** | `score_tuned` | `score_batch` |
| **Latency Column** | `latency_tuned` | `latency_batch` |
| **Analysis** | Learning curve dynamics | Performance ceiling |
| **Compute Cost** | ~10 × finetuning runs | 1 × finetuning run |

Both approaches are implemented in dedicated Colab notebooks for reproducibility:
- **Incremental**: `colab/finetune_incremental_colab.ipynb` — runs one checkpoint at a time
- **Batch**: `colab/finetune_batch_colab.ipynb` — finetunes on full 50K in a single run

#### Extended Evaluation Metrics

Phase 2 extends the 7-metric evaluation suite with additional text overlap metrics:

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1-7 | Core 7 Metrics | Model-Agnostic | See [Evaluation Framework](#evaluation-framework) |
| 8 | **ROUGE-1** | Token Overlap | Unigram overlap with teacher reference |
| 9 | **ROUGE-L** | Longest Common Subsequence | LCS-based similarity with teacher |
| 10 | **BLEU** | n-gram Precision | Modified n-gram precision against teacher |

```python
# Extended scoring for Phase 2
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference=teacher_output, hypothesis=student_output)
bleu = sentence_bleu([teacher_output.split()], student_output.split())

# Combined score includes all 10 metrics
overall = mean(7_core_metrics) × (1 - hallucination) + rouge_bleu_bonus
```

#### Data Schema (modelcomp_50k)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT | Primary key (1-50000) |
| `checkpoint` | INT | Which 5K batch (1-10) |
| `status` | TEXT | Record lifecycle state (score/finetune/output_tuned/score_tuned/completed) |
| `input` | TEXT | Instruction/prompt |
| `context` | TEXT | Optional context |
| `sevenb` | TEXT | Teacher output (Alpaca-7B) |
| `student_output` | TEXT | Base student output (pre-finetuning) |
| `score` | DECIMAL | Base student overall score |
| `generation_latency` | DECIMAL | Base generation time (seconds) |
| `student_output_tuned` | TEXT | Finetuned model output (incremental) |
| `score_tuned` | DECIMAL | Finetuned model overall score |
| `latency_tuned` | DECIMAL | Finetuned generation latency |
| `improvement` | DECIMAL | score_tuned − score (per-record delta) |
| `student_output_batch` | TEXT | Batch-trained model output |
| `score_batch` | DECIMAL | Batch-trained overall score |
| `latency_batch` | DECIMAL | Batch-trained generation latency |

#### Colab Training Infrastructure

All Phase 2 training runs on **Google Colab T4 GPU** for reproducibility and accessibility:

| Setting | Value |
|---------|-------|
| **GPU** | NVIDIA T4 (16 GB VRAM) |
| **Install** | `pip install unsloth` (manages full dependency tree) |
| **Model Loading** | `unsloth/gemma-3-1b-it-bnb-4bit` via FastModel |
| **LoRA Targets** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Optimizer** | AdamW 8-bit |
| **Epochs per Checkpoint** | 1 (Colab) / 3 (full run) |
| **MAX_NEW_TOKENS** | 512 |
| **Model Persistence** | Google Drive mount for LoRA adapter checkpoints |
| **Database** | Supabase PostgreSQL (remote, persistent across sessions) |

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

### Phase 2: Proposed Scientific Analysis (Planned)

> **Note**: The following analysis sections are **proposed/planned** and will be populated once all 10 checkpoint runs are complete. They outline the scientific analyses we intend to perform on the incremental learning results.

#### 1. Learning Curve Across All 10 Checkpoints

Comprehensive learning curve tracking overall score as training data increases from 5K to 50K samples.

```
  Overall Score
  ↑
  │                                          ← Convergence zone?
  │                              ┌───────────────────────
  │                         ┌────┘
  │                    ┌────┘
  │               ┌────┘
  │          ┌────┘
  │     ┌────┘
  │┌────┘
  │┘
  └──────────────────────────────────────────→ Training Data
   5K   10K   15K   20K   25K   30K   35K   40K   45K   50K
   C1    C2    C3    C4    C5    C6    C7    C8    C9    C10
```

*Planned table — to be updated with actual results:*

| Checkpoint | Training Data | Train Loss | Eval Loss | Overall Score | Δ Score | Cumulative Δ |
|------------|--------------|------------|-----------|---------------|---------|--------------|
| 1 | 5K | 1.0814 | 0.9278 | 0.4883 | — | — |
| 2 | 10K | 1.0757 | 1.0042 | 0.4904 | +0.0021 | +0.0021 |
| 3 | 15K | *pending* | *pending* | *pending* | — | — |
| 4 | 20K | *pending* | *pending* | *pending* | — | — |
| 5 | 25K | *pending* | *pending* | *pending* | — | — |
| 6 | 30K | *pending* | *pending* | *pending* | — | — |
| 7 | 35K | *pending* | *pending* | *pending* | — | — |
| 8 | 40K | *pending* | *pending* | *pending* | — | — |
| 9 | 45K | *pending* | *pending* | *pending* | — | — |
| 10 | 50K | *pending* | *pending* | *pending* | — | — |

#### 2. Comprehensive Metric Evolution Table

Full 10-checkpoint evolution for all 7 core metrics + ROUGE/BLEU, enabling identification of per-metric convergence patterns.

*Planned table — to be updated with actual results:*

| Metric | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Trend |
|--------|----|----|----|----|----|----|----|----|----|----|-------|
| Structured Correctness | 0.5211 | 0.5189 | — | — | — | — | — | — | — | — | *TBD* |
| Task Success | 0.6072 | 0.6105 | — | — | — | — | — | — | — | — | *TBD* |
| Instruction Following | 0.9907 | 0.9905 | — | — | — | — | — | — | — | — | *TBD* |
| Coverage | 0.2922 | 0.2951 | — | — | — | — | — | — | — | — | *TBD* |
| Faithfulness | 0.4092 | 0.4151 | — | — | — | — | — | — | — | — | *TBD* |
| Hallucination ↓ | 0.2587 | 0.2573 | — | — | — | — | — | — | — | — | *TBD* |
| Context Grounding | 0.8226 | 0.8273 | — | — | — | — | — | — | — | — | *TBD* |
| ROUGE-1 | — | — | — | — | — | — | — | — | — | — | *TBD* |
| ROUGE-L | — | — | — | — | — | — | — | — | — | — | *TBD* |
| BLEU | — | — | — | — | — | — | — | — | — | — | *TBD* |
| **Overall Score** | **0.4883** | **0.4904** | — | — | — | — | — | — | — | — | *TBD* |

#### 3. Training Efficiency Analysis (Planned)

Planned analyses on training dynamics:

- **Loss Curves**: Train loss vs. eval loss across all 10 checkpoints — watching for overfitting signals (eval loss increasing while train loss decreases)
- **Training Time Scaling**: How cumulative training time grows as dataset size increases (expected: sub-linear due to early convergence)
- **Inference Latency**: Generation latency evolution — does finetuning affect generation speed?
- **Data Efficiency**: Score improvement per 5K additional samples — identifying the cost-benefit sweet spot

$$\text{Data Efficiency}_n = \frac{\text{Score}_{n} - \text{Score}_{n-1}}{5000 \text{ samples}}$$

#### 4. Statistical Significance Analysis (Planned)

For robust scientific claims, the following statistical tests will be applied:

- **Confidence Intervals**: 95% CI for each metric at each checkpoint using bootstrap sampling
- **Paired t-tests**: Checkpoint-to-checkpoint significance testing (H₀: no improvement)
- **Effect Sizes**: Cohen's d for practical significance beyond statistical significance
- **Multiple Comparison Correction**: Bonferroni correction for testing across 10 checkpoints

$$\text{Cohen's } d = \frac{\bar{X}_{ckpt_n} - \bar{X}_{ckpt_{n-1}}}{s_{pooled}}$$

#### 5. Diminishing Returns Analysis (Planned)

Identifying the **knee point** where additional training data yields minimal improvement:

- **Marginal Improvement**: Δ Score per checkpoint (expected to decrease)
- **Cost-Benefit Ratio**: Score gain vs. compute cost (training time + inference time)
- **Knee Identification**: Using the Kneedle algorithm or second derivative test
- **Optimal Data Size**: Recommendation for minimum data needed to achieve X% of maximum performance

$$\text{Marginal Return}_n = \frac{\Delta\text{Score}_n}{\Delta\text{Compute}_n} = \frac{\text{Score}_n - \text{Score}_{n-1}}{\text{Time}_n}$$

#### 6. Category-Wise Performance Evolution (Planned)

Per-task-category analysis across all 10 checkpoints to identify:
- Which categories benefit most from additional training data
- Whether certain categories plateau earlier than others
- Task-specific convergence patterns

*Planned table — to be updated with actual results:*

| Category | Base Score | C1 | C2 | ... | C10 | Total Δ | Best Ckpt |
|----------|-----------|----|----|-----|-----|---------|-----------|
| general_qa | *pending* | — | — | — | — | — | — |
| classification_analysis | *pending* | — | — | — | — | — | — |
| creative_generative | *pending* | — | — | — | — | — | — |
| language_editing | *pending* | — | — | — | — | — | — |
| math_logic | *pending* | — | — | — | — | — | — |
| technical_code | *pending* | — | — | — | — | — | — |

#### 7. Incremental vs. Batch Comparison Results (Planned)

Head-to-head comparison of incremental (10-checkpoint) vs. batch (single-pass) training:

| Metric | Base Student | Incremental (C10) | Batch (50K) | Winner |
|--------|-------------|-------------------|-------------|--------|
| Overall Score | *pending* | *pending* | *pending* | *TBD* |
| Training Time | — | *pending* | *pending* | *TBD* |
| Hallucination ↓ | *pending* | *pending* | *pending* | *TBD* |
| Faithfulness | *pending* | *pending* | *pending* | *TBD* |
| Coverage | *pending* | *pending* | *pending* | *TBD* |

**Research Questions**:
1. Does incremental training achieve comparable performance to batch training?
2. Is there a catastrophic forgetting effect in incremental training?
3. Does the batch model overfit more or less than the incremental approach?
4. Which approach generalizes better to under-represented task categories?

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
│   ├── 12_train_incremental.py        # 10-checkpoint incremental training (status-based)
│   ├── EVALUATION_METRICS.md          # Detailed metric documentation
│   └── README.md                       # Experiment pipeline docs
│
├── colab/                              # Google Colab notebooks
│   ├── base_student_colab.ipynb       # Generate base student (50K)
│   ├── finetune_incremental_colab.ipynb # Incremental finetuning (per checkpoint)
│   └── finetune_batch_colab.ipynb     # Batch finetuning (full 50K comparison)
│
├── sql/                                # Database schemas
│   ├── 01_schema_setup.sql            # Initial Supabase setup
│   ├── 02_schema_eval_matrix.sql      # Evaluation columns
│   ├── 03_schema_incremental_tables.sql # Incremental learning tables
│   ├── 04_schema_50k_checkpoints.sql  # 50K checkpoint columns
│   ├── 05_schema_batch_columns.sql    # Batch comparison columns
│   ├── 05_schema_incremental_pipeline.sql # Status-based pipeline schema
│   └── 06_cleanup_legacy_columns.sql  # Migration from ckpt columns to status-based
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

**Step 1: Generate Base Student Outputs**
1. Upload `colab/base_student_colab.ipynb` to Google Colab
2. Enable T4 GPU: Runtime → Change runtime type → T4 GPU
3. Add Supabase credentials (`SUPABASE_URL`, `SUPABASE_KEY`) in Cell 2
4. Run all cells — generates base student outputs for 50K records
5. Records will be updated with `student_output`, `score`, and `status='finetune'`

**Step 2: Incremental Finetuning (10 Checkpoints)**
1. Upload `colab/finetune_incremental_colab.ipynb` to Google Colab
2. Enable T4 GPU and add Supabase credentials
3. Set `CHECKPOINT = 1` in the config cell
4. Run all cells — the notebook will:
   - Fetch records where `status='finetune'` for the current checkpoint
   - Finetune Gemma 3:1B with LoRA on cumulative data
   - Generate `student_output_tuned` for checkpoint records
   - Score with 7 metrics + ROUGE/BLEU → `score_tuned`
   - Calculate `improvement` = score_tuned − score
   - Update status to `completed`
5. Increment `CHECKPOINT = 2` and repeat for checkpoints 2–10
6. LoRA adapters are saved to Google Drive for persistence

**Step 3: Batch Comparison (Optional)**
1. Upload `colab/finetune_batch_colab.ipynb` to Google Colab
2. Run all cells — trains on full 50K at once
3. Generates `student_output_batch`, `score_batch`, `latency_batch`

#### Option B: Local GPU

```bash
# Generate base student outputs
python experiment/11_gen_base_student.py

# Run incremental checkpoints (status-based workflow)
python experiment/12_train_incremental.py --checkpoint 1
python experiment/12_train_incremental.py --checkpoint 2
# ... continue for checkpoints 3-10
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
