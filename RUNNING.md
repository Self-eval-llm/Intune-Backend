# Running the LLM Self-Improvement System

## Overview
This system has 3 independent components that run continuously:

1. **API Server** (`app/app.py`) - Handles `/generate` endpoint
2. **First Evaluation Worker** (`app/eval_first.py`) - Evaluates base model outputs
3. **Fine-tune Worker** (`app/eval_finetune.py`) - Runs fine-tuning at 5000 records, then evaluates fine-tuned model

## How to Run

### Terminal 1: Start API Server
```powershell
cd C:\Users\Radhakrishna\Downloads\llm
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Start First Evaluation Worker
```powershell
cd C:\Users\Radhakrishna\Downloads\llm
python app/eval_first.py
```

### Terminal 3: Start Fine-tune Worker
```powershell
cd C:\Users\Radhakrishna\Downloads\llm
python app/eval_finetune.py
```

## Workflow

1. **Frontend calls `/generate`**
   - API generates response from Ollama
   - Inserts record to Supabase with `status_eval_first='ready'`

2. **eval_first.py worker**
   - Polls for records with `status_eval_first='ready'`
   - Computes 8 metrics using base model output
   - Updates metrics columns in Supabase
   - Sets `status_eval_first='done'`

3. **eval_finetune.py worker**
   - Monitors count of records with `status_eval_first='done'`
   - When count reaches 5000:
     - Runs `src/training/finetune.py`
     - Loads fine-tuned model
     - Evaluates all records with fine-tuned model
     - Updates `*_tuned` metric columns
     - Sets `status_eval_final='done'`

## Database Columns

### Status Columns
- `status_eval_first`: "ready" → "done" (by eval_first.py)
- `status_eval_final`: null → "done" (by eval_finetune.py)

### Metric Columns (Base Model)
- `answer_relevancy`, `contextual_precision`, `contextual_recall`, `contextual_relevancy`
- `faithfulness`, `toxicity`, `hallucination_rate`, `overall`

### Metric Columns (Fine-tuned Model)
- `answer_relevancy_tuned`, `contextual_precision_tuned`, etc.
- `actual_output_tuned` (fine-tuned model's response)

## Notes

- All metrics stored as INT8 (multiply by 10000, rounded)
- Workers run continuously until manually stopped (Ctrl+C)
- Fine-tune worker exits after completing all evaluations
- Logs show progress in real-time
