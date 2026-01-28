# Phase 2: Incremental Self-Learning Loop (50K)

## Overview

Based on the **Alpaca vs OSS-20B comparison** (Alpaca won with 57.2% wins), we now scale to **50K records** with **incremental learning** to demonstrate that the model **LEARNS and IMPROVES** with more data.

## Key Results from Phase 1

| Metric | Alpaca | OSS-20B | Winner |
|--------|--------|---------|--------|
| Faithfulness | 0.3603 | 0.2678 | **Alpaca** |
| Hallucination | 0.1814 | 0.3063 | **Alpaca** |
| Overall Score | 0.4667 | 0.3817 | **Alpaca** |

**Recommendation:** Use Alpaca as teacher for 50K training.

---

## The Self-Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-LEARNING LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   For each checkpoint (5K, 10K, 15K, ... 50K):                 │
│                                                                 │
│   1. TRAIN student on Alpaca teacher outputs                    │
│      ↓                                                          │
│   2. GENERATE student outputs on fixed eval set                 │
│      ↓                                                          │
│   3. COMPARE student vs teacher (7 metrics)                     │
│      ↓                                                          │
│   4. TRACK improvement across checkpoints                       │
│      ↓                                                          │
│   5. PROVE: More data → Better student!                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   SELF-LEARNING LOOP (50K)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   DISTRIBUTED GENERATION (Ray Cluster)                         │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│   │ LOQ 4060 │   │ MBP Pro  │   │ MBP Air  │                   │
│   │ (HEAD)   │   │ (Worker) │   │ (Worker) │                   │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                   │
│        └──────────────┼──────────────┘                         │
│                       ▼                                         │
│              ┌─────────────────┐                               │
│              │    Supabase     │                               │
│              │  modelComp_50k  │                               │
│              └────────┬────────┘                               │
│                       ▼                                         │
│   ┌────────────────────────────────────────────────────────┐   │
│   │           INCREMENTAL LEARNING LOOP                     │   │
│   │                                                         │   │
│   │   5K → 10K → 15K → 20K → 25K → 30K → 35K → 40K → 50K   │   │
│   │    ↓      ↓      ↓      ↓      ↓      ↓      ↓     ↓    │   │
│   │  eval   eval   eval   eval   eval   eval   eval  eval   │   │
│   │                                                         │   │
│   │   📈 Track: Faithfulness↑, Hallucination↓, Coverage↑    │   │
│   └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 10 | `10_ray_cluster_setup.py` | Setup Ray distributed cluster across 3 machines |
| 11 | `11_distributed_data_generation.py` | Generate student outputs using Ray workers |
| 12 | `12_incremental_finetune.py` | Incremental fine-tuning with checkpoints |
| 13 | `13_learning_curve_visualization.py` | Visualize learning curves and improvement |

---

## Quick Start

### Step 1: Prepare Alpaca 50K Dataset
```powershell
# Downloads and formats Alpaca 52K → 50K with proper structure
python experiment/12_incremental_finetune.py --prepare-data
```

This creates `data/experiment/alpaca_50k_prepared.json` with:
- `instruction`: Formatted instruction
- `input`: Context + Question formatted
- `teacher_output`: Alpaca's output (the teacher!)
- `checkpoint`: Which 5K batch (1-10)

### Step 2: Run Incremental Training
```powershell
# Dry run to see the plan
python experiment/12_incremental_finetune.py --dry-run

# Run full pipeline (5K → 50K with eval at each checkpoint)
python experiment/12_incremental_finetune.py

# Start from specific checkpoint (if resuming)
python experiment/12_incremental_finetune.py --start-checkpoint 3

# Only evaluate existing checkpoints (no training)
python experiment/12_incremental_finetune.py --eval-only
```

### Step 3: (Optional) Distributed Generation
```powershell
# If you want to generate outputs faster using Ray cluster:
python experiment/10_ray_cluster_setup.py --head  # On Lenovo LOQ
python experiment/10_ray_cluster_setup.py --worker --head-ip <IP>  # On MacBooks

# Then generate distributed:
python experiment/11_distributed_data_generation.py --checkpoint 5 --distributed
```

### Step 4: Visualize Learning Curve
```powershell
python experiment/13_learning_curve_visualization.py
python experiment/13_learning_curve_visualization.py --format pdf
```

---

## What Each Script Does

| Step | Script | Purpose |
|------|--------|---------|
| 10 | `10_ray_cluster_setup.py` | Setup Ray cluster (OPTIONAL - for speed) |
| 11 | `11_distributed_data_generation.py` | Distributed generation (OPTIONAL - for speed) |
| **12** | **`12_incremental_finetune.py`** | **MAIN SCRIPT - Full pipeline** |
| 13 | `13_learning_curve_visualization.py` | Visualize results |

---

## Expected Output

### Learning Curve Table
```
================================================================================
📈 INCREMENTAL LEARNING CURVE - Self-Learning Loop Progress
================================================================================
CP   Data     Faith      Halluc     Cover      Ground     Overall   
--------------------------------------------------------------------------------
1    5000     0.3012     0.2845     0.1856     0.8012     0.3521    
2    10000    0.3245 ↑   0.2654 ↓   0.2012 ↑   0.8123 ↑   0.3789 ↑  
3    15000    0.3478 ↑   0.2512 ↓   0.2145 ↑   0.8234 ↑   0.4012 ↑  
...
10   50000    0.4523 ↑   0.1756 ↓   0.2756 ↑   0.8723 ↑   0.5234 ↑  
--------------------------------------------------------------------------------

🎯 IMPROVEMENT SUMMARY (Checkpoint 1 → 10):
   faithfulness        : +0.1511 ✅ Improved
   hallucination       : -0.1089 ✅ Improved
   overall_score       : +0.1713 ✅ Improved

💡 KEY INSIGHT: Model shows self-learning improvement!
```

### Key Insight

> **"The model demonstrates SELF-LEARNING where each additional 5K records 
> improves faithfulness, reduces hallucination, and increases overall score."**

---

## Hardware Requirements

| Machine | Role | Specs |
|---------|------|-------|
| Lenovo LOQ | HEAD + GPU Training | RTX 4060, 16GB RAM |
| MacBook Pro | Worker (Generation) | M1/M2, 16GB RAM |
| MacBook Air | Worker (Generation) | M1/M2, 8GB RAM |

---

## Files Created

```
experiment/
├── 10_ray_cluster_setup.py          # Ray cluster management
├── 11_distributed_data_generation.py # Distributed generation
├── 12_incremental_finetune.py       # Incremental training loop
├── 13_learning_curve_visualization.py # Visualization

sql/
├── 10_incremental_learning_tables.sql # Supabase schema

models/
├── incremental_checkpoints/
│   ├── checkpoint_1/                # 5K model
│   ├── checkpoint_2/                # 10K model
│   └── ...

reports/
├── incremental_learning/
│   ├── incremental_learning_results.json
│   └── incremental_learning_results.csv
├── visualizations/
│   ├── learning_curves.png
│   ├── improvement_heatmap.png
│   └── learning_summary.md
```

---

## Metrics Tracked

| Metric | Description | Goal |
|--------|-------------|------|
| Faithfulness | How well student follows teacher | ↑ Increase |
| Hallucination | Ungrounded content ratio | ↓ Decrease |
| Coverage | Completeness vs teacher | ↑ Increase |
| Context Grounding | Use of context terms | ↑ Increase |
| Overall Score | Weighted combination | ↑ Increase |

---

## Troubleshooting

### Ray Connection Issues
```powershell
# Check if Ray is running
python experiment/10_ray_cluster_setup.py --status

# Firewall: Allow port 6379, 8265
```

### Ollama Not Found
```powershell
# Ensure Ollama is running on all machines
ollama serve

# Check model is available
ollama list
```

### Memory Issues
```powershell
# Reduce batch size in 12_incremental_finetune.py
BATCH_SIZE = 2  # Instead of 4
```
