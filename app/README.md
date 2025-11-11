# API Documentation

This directory contains the FastAPI applications for the LLM evaluation system.

## 📁 Files Overview

- **`app.py`** - 🎯 **UNIFIED API SERVER** - All endpoints (Generate, Finetune, Evaluation)
- **`finetune_worker.py`** - Finetune logic module (imported by app.py)
- **`eval_worker.py`** - Evaluation logic module (imported by app.py)
- **`generate.py`** - Ollama model integration utilities
- **`evaluateapi.py`** - Legacy compatibility layer (deprecated)
- **`__init__.py`** - Package initialization

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements_complete.txt
```

### Environment Setup

Create a `.env` file in the project root:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

### Starting the Unified API Server

```bash
# 🎯 Start EVERYTHING with one command (Port 8000)
python app/app.py

# This single server provides ALL functionality:
# ✅ Model Generation (/generate)
# ✅ Finetune Automation (/finetune/*)  
# ✅ Evaluation Workers (/eval/*)
```

---

## 🎯 **UNIFIED API SERVER** (Port 8000)

**🚀 Single Command:** `python -m app.app`  
**Base URL:** `http://localhost:8000`  
**Features:** Model Generation + Finetune Automation + Evaluation Workers

---

## 🤖 Model Generation Endpoints

### Endpoints

#### `GET /`
**Description:** Root endpoint providing API information

**Response:**
```json
{
  "message": "Gemma Model API",
  "version": "1.0.0",
  "model": "gemma-finetuned"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

#### `GET /health`
**Description:** Health check endpoint to monitor service status

**Response:**
```json
{
  "status": "healthy",
  "supabase_connection": true,
  "finetune_worker": {
    "running": true,
    "last_check": "2025-11-07T10:30:00",
    "conditions_met": false,
    "total_rows": 3500,
    "completed_evaluations": 2800
  },
  "ollama_running": true,
  "model_loaded": true,
  "model_name": "gemma-finetuned"
}
```

**Status Values:**
- `healthy` - All services operational
- `degraded` - Some issues detected

**Example:**
```bash
curl http://localhost:8000/health
```

---

#### `POST /generate`
**Description:** Generate a response from the fine-tuned Gemma model

**Request Body:**
```json
{
  "prompt": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is a method of data analysis that automates analytical model building...",
  "model": "gemma-finetuned"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain neural networks"}'
```

**Error Responses:**
- `503` - Service unavailable (Ollama not running or model not loaded)
- `500` - Internal server error during generation

---

### Finetune Automation Endpoints

#### `POST /finetune`
**Description:** Start background worker to monitor and trigger fine-tuning automatically

**Conditions for Auto-Execution:**
- Total rows in `intune_db` >= 5000
- Rows with `status_eval_final = true` >= 5000

**Response:**
```json
{
  "success": true,
  "message": "Finetune worker started successfully. It will monitor conditions every 5 minutes.",
  "worker_running": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/finetune
```

---

#### `GET /finetune/status`
**Description:** Check status of finetune worker and conditions

**Response:**
```json
{
  "worker_running": true,
  "last_check": "2025-11-07T10:30:00",
  "conditions_met": false,
  "total_rows": 3500,
  "completed_evaluations": 2800
}
```

**Example:**
```bash
curl http://localhost:8000/finetune/status
```

---

#### `POST /finetune/stop`
**Description:** Stop the finetune background worker

**Response:**
```json
{
  "success": true,
  "message": "Finetune worker stopped successfully",
  "worker_running": false
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/finetune/stop
```

---

#### `POST /finetune/run-now`
**Description:** Execute fine-tuning immediately (bypasses condition checks)

**⚠️ Warning:** Use with caution - ensure sufficient data exists before manual execution.

**Response:**
```json
{
  "success": true,
  "message": "Fine-tuning completed successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/finetune/run-now
```

---

## 🔄 Evaluation Worker API (Port 8001)

**Base URL:** `http://localhost:8001`

This API provides background workers that continuously monitor the Supabase `intune_db` table and automatically process evaluation tasks.

### Status & Monitoring Endpoints

#### `GET /`
**Description:** Root endpoint providing API information

**Response:**
```json
{
  "message": "Evaluation Worker API",
  "version": "1.0.0",
  "description": "Background workers for continuous evaluation monitoring"
}
```

---

#### `GET /status`
**Description:** Get current status of background workers

**Response:**
```json
{
  "first_eval_worker": true,
  "final_eval_worker": true,
  "uptime_seconds": 3600
}
```

**Example:**
```bash
curl http://localhost:8001/status
```

---

#### `GET /health`
**Description:** Health check for API and database connectivity

**Response:**
```json
{
  "status": "healthy",
  "supabase_connection": true,
  "workers": {
    "first_eval": true,
    "final_eval": true
  },
  "uptime_seconds": 3600
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Database or worker issues

**Example:**
```bash
curl http://localhost:8001/health
```

---

#### `GET /metrics/pending`
**Description:** Get count of pending evaluations in the database

**Response:**
```json
{
  "pending_first_evaluations": 25,
  "pending_final_evaluations": 10,
  "total_pending": 35
}
```

**Example:**
```bash
curl http://localhost:8001/metrics/pending
```

### Worker Control Endpoints

#### `POST /start-all-workers`
**Description:** Start both evaluation workers

**Response:**
```json
{
  "message": "All workers started",
  "details": [
    "First evaluation worker started",
    "Final evaluation worker started"
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/start-all-workers
```

---

#### `POST /stop-all-workers`
**Description:** Stop both evaluation workers

**Response:**
```json
{
  "message": "All workers stopped",
  "details": [
    "First evaluation worker stopped",
    "Final evaluation worker stopped"
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/stop-all-workers
```

---

#### `POST /start-first-eval-worker`
**Description:** Start only the first evaluation worker (base model metrics)

**Response:**
```json
{
  "message": "First evaluation worker started successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/start-first-eval-worker
```

---

#### `POST /stop-first-eval-worker`
**Description:** Stop only the first evaluation worker

**Response:**
```json
{
  "message": "First evaluation worker stopped successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/stop-first-eval-worker
```

---

#### `POST /start-final-eval-worker`
**Description:** Start only the final evaluation worker (fine-tuned model metrics)

**Response:**
```json
{
  "message": "Final evaluation worker started successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/start-final-eval-worker
```

---

#### `POST /stop-final-eval-worker`
**Description:** Stop only the final evaluation worker

**Response:**
```json
{
  "message": "Final evaluation worker stopped successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/stop-final-eval-worker
```

---

## 🔧 Worker Functionality

### First Evaluation Worker
- **Monitors:** Records where `status_eval_first = False`
- **Function:** Computes base model metrics using existing evaluation logic
- **Updates:** Base metric columns (answer_relevancy, faithfulness, etc.)
- **Completion:** Sets `status_eval_first = True`
- **Processing Rate:** ~10 records per batch, checks every 5 seconds

### Final Evaluation Worker  
- **Monitors:** Records where `status_eval_final = False` AND `status_eval_first = True`
- **Function:** Loads fine-tuned model, generates outputs, computes metrics
- **Updates:** Fine-tuned metric columns (*_tuned suffix)
- **Completion:** Sets `status_eval_final = True`
- **Processing Rate:** ~5 records per batch, checks every 10 seconds

---

## 📊 Database Integration

### Required Table Structure (`intune_db`)

```sql
-- Core data columns
input TEXT,                    -- Question/prompt
expected_output TEXT,          -- Reference answer
context JSONB,                 -- Context information
actual_output TEXT,            -- Base model output

-- Status columns
status_eval_first BOOLEAN DEFAULT FALSE,
status_eval_final BOOLEAN DEFAULT FALSE,

-- Base model metrics (INT8: multiply by 10000)
answer_relevancy INTEGER,
contextual_precision INTEGER,
contextual_recall INTEGER,
contextual_relevancy INTEGER,
faithfulness INTEGER,
toxicity INTEGER,
hallucination_rate INTEGER,
overall INTEGER,

-- Fine-tuned model metrics
actual_output_tuned TEXT,
answer_relevancy_tuned INTEGER,
contextual_precision_tuned INTEGER,
contextual_recall_tuned INTEGER,
contextual_relevancy_tuned INTEGER,
faithfulness_tuned INTEGER,
toxicity_tuned INTEGER,
hallucination_rate_tuned INTEGER,
overall_tuned INTEGER
```

### Sample Data Insertion

```sql
INSERT INTO intune_db (
    input,
    expected_output,
    context,
    actual_output,
    status_eval_first,
    status_eval_final
) VALUES (
    'What is machine learning?',
    'Machine learning is automated analytical model building.',
    '["ML is subset of AI", "Uses algorithms to analyze data"]',
    'Machine learning helps computers learn from data.',
    False,  -- Triggers first evaluation
    False   -- Triggers final evaluation
);
```

---

## 🧪 Testing

### Quick API Test

```bash
# Test both APIs
python test_evaluation_api.py

# Individual tests
curl http://localhost:8000/health    # Gemma API
curl http://localhost:8001/health    # Evaluation API
```

### Manual Testing Sequence

```bash
# 1. Check API status
curl http://localhost:8001/status

# 2. Check pending work
curl http://localhost:8001/metrics/pending

# 3. Stop workers
curl -X POST http://localhost:8001/stop-all-workers

# 4. Start workers
curl -X POST http://localhost:8001/start-all-workers

# 5. Generate with Gemma
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI"}'
```

---

## 📈 Performance Characteristics

### Gemma Model API
- **Latency:** 2-5 seconds per generation (depends on prompt length)
- **Throughput:** ~10-20 requests/minute
- **Memory:** ~2-4GB VRAM for model inference

### Evaluation Worker API
- **Base Evaluation:** ~10-20 records/minute (CPU-bound)
- **Fine-tuned Evaluation:** ~2-5 records/minute (GPU-bound)
- **Memory:** ~4-6GB VRAM when fine-tuned model loaded

---

## 🛠️ Configuration

### Environment Variables

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Optional
TORCHDYNAMO_DISABLE=1              # Disable torch compilation
TORCHINDUCTOR_COMPILE_THREADS=1    # Single-threaded compilation
```

### Model Configuration

Edit `generate.py` for Ollama model settings:
```python
MODEL_NAME = "gemma-finetuned"     # Your model name
OLLAMA_HOST = "localhost:11434"    # Ollama server
```

Edit `evaluateapi.py` for worker settings:
```python
# First worker batch size
.limit(10)  # Increase for faster processing

# Final worker batch size  
.limit(5)   # Reduce if memory issues
```

---

## 🚨 Error Handling

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Check if running
   curl http://localhost:11434/api/tags
   ```

2. **Model Not Found**
   ```bash
   # Create model from Modelfile
   ollama create gemma-finetuned -f Modelfile
   ```

3. **Supabase Connection Failed**
   - Verify credentials in `.env`
   - Check network connectivity
   - Ensure table `intune_db` exists

4. **Workers Not Processing**
   - Check database has records with `status_eval_first=False`
   - Verify fine-tuned model exists at `models/gemma-finetuned-merged/`
   - Check worker logs for specific errors

### Recovery Procedures

```bash
# Restart workers
curl -X POST http://localhost:8001/stop-all-workers
# Wait 30 seconds
curl -X POST http://localhost:8001/start-all-workers

# Check status
curl http://localhost:8001/health
```

---

## 📋 API Summary

| Endpoint | Method | Port | Purpose |
|----------|--------|------|---------|
| `/` | GET | 8000/8001 | API information |
| `/health` | GET | 8000/8001 | Health check |
| `/generate` | POST | 8000 | Generate model response |
| `/finetune` | POST | 8000 | Start finetune worker |
| `/finetune/status` | GET | 8000 | Check finetune status |
| `/finetune/stop` | POST | 8000 | Stop finetune worker |
| `/finetune/run-now` | POST | 8000 | Execute finetune manually |
| `/status` | GET | 8001 | Worker status |
| `/metrics/pending` | GET | 8001 | Pending evaluations |
| `/start-all-workers` | POST | 8001 | Start both workers |
| `/stop-all-workers` | POST | 8001 | Stop both workers |
| `/start-first-eval-worker` | POST | 8001 | Start base eval worker |
| `/stop-first-eval-worker` | POST | 8001 | Stop base eval worker |
| `/start-final-eval-worker` | POST | 8001 | Start final eval worker |
| `/stop-final-eval-worker` | POST | 8001 | Stop final eval worker |

---

## 🔗 Related Files

- **Main Project:** [../README.md](../README.md)
- **Evaluation API Docs:** [../EVALUATION_API.md](../EVALUATION_API.md)
- **Requirements:** [../requirements_complete.txt](../requirements_complete.txt)
- **Test Script:** [../test_evaluation_api.py](../test_evaluation_api.py)
- **Startup Script:** [../run_evaluation_api.py](../run_evaluation_api.py)