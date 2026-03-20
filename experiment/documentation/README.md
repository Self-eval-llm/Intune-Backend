# INTUNE Experiment Pipeline

## Overview

This folder contains numbered scripts for the complete experiment pipeline across two phases.

---

## Phase 1: Teacher Comparison (4K Dataset)

**Research Question:** Which teacher produces a better fine-tuned Gemma 3:1B student: Alpaca (7B) or GPT-OSS 20B?

**Result:** ✅ **Alpaca won with 57.2% win rate**

| Step | Script | Status | Description |
|------|--------|--------|-------------|
| 1 | `01_data_download_alpaca.py` | ✅ Done | Download Stanford Alpaca dataset |
| 2 | `02_data_prepare_4k.py` | ✅ Done | Prepare 4K samples for experiment |
| 3 | `03_gen_base_gemma.py` | ✅ Done | Generate Gemma outputs → Supabase |
| 4a | `04a_train_finetune_alpaca.py` | ✅ Done | Fine-tune with Alpaca teacher |
| 4b | `04b_gen_teacher_oss20b.py` | ✅ Done | Generate OSS 20B outputs |
| 5 | `05_data_label.py` | ✅ Done | Label dataset |
| 6 | `06_eval_metrics.py` | ✅ Done | Compute evaluation metrics |
| 6a | `06a_gen_tuned_alpaca.py` | ✅ Done | Generate tuned model outputs |
| 7 | `07_eval_compare_teachers.py` | ✅ Done | Compare teachers → **Alpaca wins** |
| 8 | `08_gen_context.py` | ✅ Done | Generate context |
| 9 | `09_report_analytical.py` | ✅ Done | Generate analytical report |

---

## Phase 2: Incremental Learning (50K Dataset)

**Goal:** Train student model progressively on 5K → 10K → ... → 50K records and measure improvement at each stage.

| Step | Script | Status | Description |
|------|--------|--------|-------------|
| 10 | `10_data_upload_50k.py` | ✅ Done | Upload 50K to Supabase `modelcomp_50k` |
| 11 | `11_gen_base_student.py` | 🔄 10% | Generate base student outputs (before finetuning) |
| 12 | `12_train_incremental.py` | ⏳ Pending | Run 10 incremental learning stages |

---

## Supabase Tables

### `modelComp` (Phase 1 - 4K records)

| Column | Description |
|--------|-------------|
| `id` | UUID (auto-generated) |
| `input` | Original instruction |
| `context` | Context if available |
| `actual_output` | Gemma 3:1B output |
| `sevenb` | Alpaca outputs |
| `twentyb` | GPT-OSS 20B outputs |

### `modelcomp_50k` (Phase 2 - 50K records)

| Column | Description |
|--------|-------------|
| `id` | Integer (auto-generated) |
| `input` | Instruction/prompt |
| `context` | Optional context |
| `sevenb` | Teacher output (Alpaca) |
| `student_output` | Base student (before finetuning) |
| `student_output_ckpt1-10` | Output after each stage |
| `score_ckpt1-10` | Similarity score per stage |
| `latency_ckpt1-10` | Generation latency per stage |

---

## Running Phase 2

### Option 1: Local GPU
```bash
# Step 1: Generate base student outputs
python experiment/11_gen_base_student.py

# Step 2: Run incremental stages (one at a time)
python experiment/12_train_incremental.py --stage 1
python experiment/12_train_incremental.py --stage 2
# ... continue for stages 3-10
```

### Option 2: Google Colab (Recommended)
1. Upload `colab/base_student_colab.ipynb` to Colab
2. Enable T4 GPU
3. Add Supabase credentials
4. Run all cells
5. Then upload `colab/finetune_incremental_colab.ipynb` for each stage

---

## Results Location

| File | Description |
|------|-------------|
| `reports/teacher_comparison_report.json` | Phase 1 comparison results |
| `reports/incremental_learning/` | Phase 2 stage-by-stage results |

### Step 5: Fine-tune with Alpaca
```powershell
python experiment/05_finetune_with_alpaca.py --epochs 3
```

### Step 6: Fine-tune with OSS 20B
```powershell
python experiment/06_finetune_with_oss20b.py --epochs 3
```

### Step 7: Evaluate & Compare
```powershell
python experiment/07_evaluate_compare_teachers.py --samples 500
```

---

## Expected Output (Step 7)

```
📊 Metric Comparison:
--------------------------------------------------------------------------------
Metric                    Alpaca       OSS-20B      p-value    Winner    
--------------------------------------------------------------------------------
answer_relevancy          0.7823       0.8012       0.0234     OSS-20B*
contextual_precision      0.6543       0.6821       0.1234     Tie
faithfulness              0.8234       0.8456       0.0056     OSS-20B*
...
--------------------------------------------------------------------------------

🏆 WINNER: OSS-20B
   Alpaca wins: 2 metrics
   OSS-20B wins: 5 metrics

✅ Recommendation: Use OSS-20B as teacher for full 50K training
```

---

## After Experiment

Once winner is declared:
1. Generate full 50K dataset with winning teacher
2. Fine-tune Gemma with full 50K (batch training)
3. Compare batch vs incremental training strategies
4. Final evaluation and paper/report
