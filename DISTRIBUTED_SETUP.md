# Distributed Output Generation Setup

## Overview

This setup allows you to use multiple machines (Windows RTX + MacBook Pro) in parallel to generate tuned outputs much faster. 

**Expected Performance:**
- Single machine (RTX 4060): ~5 records/min = ~1000 records in ~200 minutes
- Dual machines (RTX + Mac): ~10 records/min = ~1000 records in ~100 minutes  
- **2x speedup with distributed inference!**

## Architecture

```
Windows RTX (Main)
├── Runs main script (12_train_incremental.py)
├── Hosts Supabase (cloud database)
├── Monitors progress
└── Optional: Worker (13_distributed_worker.py)

MacBook Pro (Worker)
├── Runs worker (13_distributed_worker.py)
├── Pulls batches from Supabase
├── Generates outputs using 4-bit model
├── Updates results in Supabase
└── Tracks progress
```

Both machines:
- Share Supabase database credentials (.env)
- Process records in parallel without collisions
- Report progress every 10 seconds
- Show estimated completion time

## Setup Instructions

### Step 1: Prepare the Model (Windows RTX)

After finetuning completes:

```bash
# Model trained and saved at:
# models/gemma-ckpt1-lora/

# Copy to Google Drive or cloud storage to share with MacBook
# Example: drag models/gemma-ckpt1-lora to Google Drive
```

### Step 2: Setup MacBook Pro

```bash
# 1. Clone your repo
git clone <your-repo>
cd Intune_Backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with Supabase credentials
cp .env.example .env
# Edit .env and add your SUPABASE_URL and SUPABASE_KEY

# 5. Download trained model from Google Drive
# Save to models/gemma-ckpt1-lora/

# 6. Verify model is loaded correctly
ls models/gemma-ckpt1-lora/
# Should see: adapter_config.json, adapter_model.bin, config.json, tokenizer.json, etc.
```

### Step 3: Start Distributed Generation on Windows (Main)

```bash
# Terminal 1: Start progress monitor
cd c:\Users\Radhakrishna\Desktop\Intune\Intune_Backend
venv\Scripts\activate.ps1
python experiment/phase2_incremental/12_train_incremental.py --checkpoint 1 --step output_tuned
```

This will:
- Validate 4936 records ready for generation
- Display instructions for running workers
- Start monitoring dashboard
- Show real-time progress from both machines

### Step 4: Start Worker on MacBook

```bash
# On MacBook - Terminal 2
source venv/bin/activate
python experiment/phase2_incremental/13_distributed_worker.py --checkpoint 1 --worker-id mac-1

# Wait for model to load (~30 seconds)
# Then you should see:
# "DISTRIBUTED WORKER: mac-1"
# "Model loaded on: Metal GPU"
```

### Step 5: Optional - Start Second Worker on Windows

```bash
# On Windows - Terminal 3 (if you want both machines processing):
venv\Scripts\activate.ps1
python experiment/phase2_incremental/13_distributed_worker.py --checkpoint 1 --worker-id rtx-1
```

## Monitoring Progress

### Main Terminal (Windows) - Progress Dashboard

```
──────────────────────────────────────────────────────────────────────
PROGRESS UPDATE - 14:35:42
──────────────────────────────────────────────────────────────────────
Completed:  842 / 4936 (17.0%)
Pending:   4094

[xxxxxxxxx..................................................................]

Worker contributions:
  mac-1     :  512 records (60.8%)
  rtx-1     :  330 records (39.2%)

Avg time per record: 2.15s
Est. completion: 15:26:33 (51m 22s remaining)
──────────────────────────────────────────────────────────────────────
```

### Worker Terminal (MacBook) - Batch Processing

```
──────────────────────────────────────────────────────────────────────────
PROGRESS DASHBOARD - Batch 5
──────────────────────────────────────────────────────────────────────────
Total records: 4936
Completed: 842 | Pending: 4094
Progress: 17.0%

[xxxxxxxxxxxxxxxxx..............................................................]

Worker contributions:
  mac-1: 512 records

Active workers: rtx-1

Processing rate: 0.47 records/sec
Est. completion: 15:26:33 (51m 22s remaining)
──────────────────────────────────────────────────────────────────────────
```

## Features

### Collision Avoidance
- Each record marked with `processing_worker` while being processed
- Multiple workers won't process the same record twice
- Failed records are unmarked and available for retry

### Progress Tracking
- Main script tracks all workers
- Workers report which machine generated each output
- `tuned_worker` column shows worker ID for each record
- Progress dashboard updates every 10 seconds

### Time Estimates
- Real-time calculation based on actual processing speed
- Assumes 2 workers processing in parallel
- Adjusts as speed changes
- Shows ETA and countdown timer

### Network Robust
- All coordination via Supabase (cloud database)
- No direct machine-to-machine communication needed
- Works over WiFi or internet
- Can disconnect/reconnect without issues

## Troubleshooting

### Worker Reports "Model not found"
```
Solution: Copy models/gemma-ckpt1-lora from Windows to MacBook
- Download from Google Drive
- Extract to: Intune_Backend/models/gemma-ckpt1-lora/
```

### Worker hangs on "Generating"
```
Solution: Restart worker
- Press Ctrl+C to stop
- Check if model loaded correctly
- Verify Metal GPU detected: "Model loaded on: Metal GPU"
```

### Duplicate records processed
```
This shouldn't happen with our collision avoidance, but if it does:
- Run: python experiment/phase2_incremental/12_train_incremental.py --checkpoint 1 --step status
- Records should show tuned_worker names
- Can manually clean up if needed
```

### Slow performance on Mac
```
Check:
- Metal GPU enabled: "python -c 'import torch; print(torch.backends.mps.is_available())'"
- Model is in 4-bit: Should see "load_in_4bit=True"
- Reduce batch_size: --batch-size 5
```

## Commands Reference

```bash
# Check progress without running worker
python 12_train_incremental.py --checkpoint 1 --step status

# Run just one worker
python 13_distributed_worker.py --checkpoint 1 --worker-id mac-1

# Run with smaller batches (slower but safer)
python 13_distributed_worker.py --checkpoint 1 --worker-id mac-1 --batch-size 5

# Run on Windows RTX as second worker
python 13_distributed_worker.py --checkpoint 1 --worker-id rtx-1
```

## Next Steps After Completion

Once all outputs generated (`output_tuned` complete):

```bash
# Score tuned outputs
python experiment/phase2_incremental/12_train_incremental.py --checkpoint 1 --step score_tuned

# Calculate improvement
python experiment/phase2_incremental/12_train_incremental.py --checkpoint 1 --step completed

# Or run all remaining steps
python experiment/phase2_incremental/12_train_incremental.py --checkpoint 1 --run-all
```

## Performance Tips

1. **Start workers quickly** - They'll sync immediately via Supabase
2. **Monitor main terminal** - Progress updates every 10 seconds
3. **Don't restart workers** - They'll pick up where they left off
4. **Use WiFi** - All communication through cloud (Supabase)
5. **Keep models synced** - Both machines should have identical model files

## Architecture Details

### Database Columns Used
- `processing_worker`: Worker currently processing record (collision avoidance)
- `processing_at`: Timestamp when worker started
- `tuned_worker`: Worker that completed generation  
- `tuned_at`: Timestamp when generation finished
- `student_output_tuned`: Generated output
- `latency_tuned`: Generation time in ms
- `status`: Current step ('output_tuned' → 'score_tuned')

### Worker Batch Processing
```
Worker Loop:
1. Fetch batch of 10 unprocessed records
2. Mark as processing_worker = "mac-1"
3. Load model (once per worker start)
4. Process each record:
   - Generate output (~2-3 seconds each)
   - Update database with results
   - Clear processing_worker
5. Report batch complete
6. Loop back to step 1
```

### Progress Updates
```
Main Script:
1. Query Supabase for status='score_tuned' count
2. Calculate completion percentage
3. Count contributions per worker
4. Estimate time remaining
5. Display dashboard
6. Sleep 10 seconds
7. Loop
```

---

**Questions?** Check the main script comments or see `12_train_incremental.py` functions:
- `step_output_tuned_distributed()` - Main coordinator
- `get_worker_progress()` - Query progress
- `estimate_time_remaining()` - Time calculation
