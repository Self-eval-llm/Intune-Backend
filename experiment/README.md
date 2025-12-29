# Teacher Comparison Experiment

## Research Question
**Which teacher produces a better fine-tuned Gemma 3:1B student: Alpaca (text-davinci-003) or GPT-OSS 20B?**

---

## Pipeline Checklist

| Step | Script | Status | Description |
|------|--------|--------|-------------|
| 1 | `01_download_alpaca.py` | ✅ Done | Download Stanford Alpaca dataset |
| 2 | `02_prepare_4k_dataset.py` | ✅ Done | Prepare 4K samples for experiment |
| 3 | `03_generate_gemma_save_supabase.py` | ✅ Done | Generate Gemma outputs → Supabase |
| 4 | `04_generate_gpt_oss_supabase.py` | ⏳ Pending | Generate OSS 20B outputs (MacBook) |
| 5 | `05_finetune_with_alpaca.py` | ⏳ Pending | Fine-tune Gemma with Alpaca teacher |
| 6 | `06_finetune_with_oss20b.py` | ⏳ Pending | Fine-tune Gemma with OSS 20B teacher |
| 7 | `07_evaluate_compare_teachers.py` | ⏳ Pending | Evaluate both & declare winner |

---

## Supabase Table: `modelComp`

| Column | Description |
|--------|-------------|
| `id` | UUID (auto-generated) |
| `input` | Original instruction from Alpaca |
| `context` | Context if available, else NULL |
| `actual_output` | Gemma 3:1B output |
| `sevenb` | Alpaca outputs (text-davinci-003) |
| `twentyb` | GPT-OSS 20B outputs |
| `created_at` | Auto timestamp |
| `updated_at` | Auto timestamp |

---

## Commands to Run

### Step 4: Generate OSS 20B outputs (on MacBook)
```bash
# On MacBook first:
OLLAMA_HOST=0.0.0.0 ollama serve

# Get IP:
ipconfig getifaddr en0

# On Windows:
python experiment/04_generate_gpt_oss_supabase.py --ip <MACBOOK_IP>
```

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
