# Self-Improving LLM Framework - Complete Architecture Guide

## 🎯 Overview

This is a **self-improving Large Language Model (LLM) evaluation framework** that automatically generates training data, evaluates model performance, fine-tunes the model, and measures improvements. The system operates continuously, creating a feedback loop that progressively enhances the model's capabilities.

### What Makes This "Self-Improving"?

The framework creates a continuous improvement cycle:
1. **Generate**: A powerful "teacher" model creates high-quality training examples
2. **Evaluate**: The base model answers questions and gets scored on 8 metrics
3. **Learn**: When enough data is collected, the model fine-tunes itself
4. **Compare**: The system measures improvements by comparing before/after metrics
5. **Repeat**: The cycle continues, progressively improving performance

---

## 🏗️ System Architecture

The framework consists of **THREE interconnected components**:

### 1️⃣ **Backend API (Intune-Backend)** - This Repository
The core processing engine that handles:
- Model inference (generating responses)
- Evaluation pipeline (computing quality metrics)
- Fine-tuning orchestration (training improved models)
- Data management and persistence

### 2️⃣ **Frontend Application** (Separate Repository)
The user interface that:
- Provides a chat interface for users to interact with the model
- Displays evaluation metrics and improvements
- Visualizes model performance over time
- Triggers manual fine-tuning when needed

**Connection**: Frontend connects to Backend via REST API on `http://localhost:8000`

### 3️⃣ **Supabase Database** (Cloud Service)
The persistent data layer that:
- Stores all questions, answers, and context
- Maintains evaluation metrics for base and fine-tuned models
- Tracks processing status for each record
- Enables resume capability if processes are interrupted

**Connection**: Backend connects to Supabase via PostgreSQL REST API

---

## 🔄 Complete System Architecture

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                    SELF-IMPROVING LLM FRAMEWORK                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

Component 1: FRONTEND                    Component 2: BACKEND
┌──────────────────────┐                ┌──────────────────────┐
│                      │                │                      │
│   User Interface     │    HTTP/REST   │   FastAPI Server     │
│   (React/Vue)        │◄──────────────►│   (Port 8000)        │
│                      │                │                      │
│  - Chat UI           │                │  - /generate         │
│  - Metrics Dashboard │                │  - /finetune         │
│  - Progress Display  │                │  - /health           │
│                      │                │                      │
└──────────────────────┘                └──────────────────────┘
                                                  │
                                                  │
                                                  ▼
                                        ┌──────────────────────┐
                                        │  Evaluation Workers   │
                                        │  (Port 8001)         │
                                        │                      │
                                        │  Worker 1:           │
                                        │  eval_first.py       │
                                        │  (Base Metrics)      │
                                        │                      │
                                        │  Worker 2:           │
                                        │  eval_finetune.py    │
                                        │  (Training + Final)  │
                                        │                      │
                                        └──────────────────────┘
                                                  │
                                                  │
                    ┌─────────────────────────────┴─────────────────────────────┐
                    │                                                           │
                    ▼                                                           ▼
Component 3: DATABASE                                            ┌──────────────────────┐
┌──────────────────────┐                                        │   Ollama Server      │
│                      │                                        │   (Port 11434)       │
│   Supabase Cloud     │                                        │                      │
│   (PostgreSQL)       │                                        │  - Gemma 1B Base     │
│                      │                                        │  - Gemma Fine-tuned  │
│  Table: intune_db    │                                        │  - GPT-OSS 20B       │
│                      │                                        │                      │
│  - Questions         │                                        └──────────────────────┘
│  - Answers           │
│  - Metrics (Base)    │
│  - Metrics (Tuned)   │
│  - Status Flags      │
│                      │
└──────────────────────┘
```

### Data Flow Sequence

```
1. User Question → Frontend
2. Frontend → POST /generate → Backend API
3. Backend API → Ollama → Generate Response
4. Backend API → Insert to Supabase (status: 'created')
5. eval_first.py Worker → Poll Supabase → Compute Base Metrics → Update DB (status: 'done')
6. eval_finetune.py Worker → Monitor Record Count → Trigger Training at Threshold
7. Fine-tuning → Train LoRA Adapters → Create Improved Model
8. eval_finetune.py → Load Fine-tuned Model → Re-evaluate All Records → Update DB
9. Frontend → Query Metrics → Display Improvements
```

### Database Status Flow

```
Initial Record:
  status_eval_first = 'created'
  status_eval_final = NULL
  (Record ready for base evaluation)

After Base Evaluation:
  status_eval_first = 'done'
  status_eval_final = NULL
  (Metrics computed, ready for fine-tuning)

After Fine-tuning Evaluation:
  status_eval_first = 'done'
  status_eval_final = 'done'
  (Complete cycle finished)
```

---

## 📊 Detailed Workflow

### Phase 1: User Interaction & Initial Response

**Step 1: User asks a question**
```
Frontend (React/Vue) → POST /generate → Backend API
```
- User types: "What is machine learning?"
- Frontend sends HTTP POST request with the prompt

**Step 2: Backend generates response**
```
Backend → Ollama (Local LLM Server) → Gemma Model
```
- Backend calls Ollama API running locally
- Gemma 1B model (or fine-tuned variant) generates response
- Response generated in 2-5 seconds

**Step 3: Save to database**
```
Backend → Supabase → intune_db table
```
Record inserted with:
```json
{
  "input": "What is machine learning?",
  "actual_output": "Machine learning is a subset of AI...",
  "status_eval_first": "created",  // Triggers evaluation
  "status_eval_final": null,
  "context": [],
  "expected_output": null
}
```

**Step 4: Return to user**
```
Backend → Frontend → Display in chat
```
- User sees the response immediately
- Background evaluation begins automatically

---

### Phase 2: First Evaluation (Base Model Metrics)

**Worker: `eval_first.py` (runs continuously)**

**Step 1: Poll for new records**
```python
# Every 5-30 seconds, check for records needing evaluation
SELECT * FROM intune_db 
WHERE status_eval_first = 'created' 
LIMIT 10
```

**Step 2: Compute 8 evaluation metrics**
For each record, calculate:

1. **Answer Relevancy** (0-1): How relevant is the answer to the question?
   - Uses cosine similarity between question and answer tokens
   
2. **Contextual Precision** (0-1): Does the answer stick to provided facts?
   - Formula: `|answer ∩ context| / |answer|`
   
3. **Contextual Recall** (0-1): Does the answer cover the context?
   - Formula: `|answer ∩ context| / |context|`
   
4. **Contextual Relevancy** (0-1): Semantic alignment with context
   - Formula: `cosine(context, answer)`
   
5. **Faithfulness** (0-1): Alignment with reference answer
   - Formula: `0.6 × cos(answer, reference) + 0.4 × cos(answer, context)`
   
6. **Toxicity** (0-1): Presence of harmful language (lower is better)
   - Formula: `toxic_words / total_words`
   
7. **Hallucination Rate** (0-1): Made-up information (lower is better)
   - Formula: `1 - contextual_precision`
   
8. **Overall Score** (0-1): Combined quality metric
   - Formula: `mean(positive_metrics) × (1 - mean(negative_metrics))`

**Step 3: Update database**
```python
UPDATE intune_db
SET 
  answer_relevancy = 7500,      # 0.75 × 10000 (stored as INT8)
  contextual_precision = 8200,
  faithfulness = 6800,
  toxicity = 150,
  hallucination_rate = 1800,
  overall = 7100,
  status_eval_first = 'done'    # Mark as evaluated
WHERE id = record_id
```

**Why multiply by 10,000?**
- PostgreSQL INT8 is more efficient than DECIMAL
- Preserves 4 decimal places of precision
- Example: 0.7532 → 7532 (stored) → 0.7532 (retrieved)

---

### Phase 3: Fine-Tuning Trigger (Automatic Improvement)

**Worker: `eval_finetune.py` (runs continuously)**

**Step 1: Monitor record count**
```python
# Check every 5 minutes
SELECT COUNT(*) FROM intune_db
WHERE status_eval_first = 'done'
  AND status_eval_final IS NULL
```

**Step 2: When threshold reached (e.g., 2+ records for testing, 5000 for production)**
```
Conditions met! Starting fine-tuning process...
```

**Step 3: Prepare training data**
```python
# Fetch evaluated records from Supabase
records = fetch_from_supabase()

# Convert to training format (JSONL)
for record in records:
    training_example = {
        "instruction": "Answer accurately based on context",
        "input": record["input"],
        "output": record["expected_output"] or record["actual_output"]
    }

# Split into 80% train, 20% validation
save_to_jsonl("data/processed/train_dataset.jsonl")
save_to_jsonl("data/processed/val_dataset.jsonl")
```

**Step 4: Execute fine-tuning**
```python
# Runs src/training/finetune.py
python src/training/finetune.py
```

This process:
- Loads base Gemma 1B model
- Applies LoRA (Low-Rank Adaptation) adapters
- Trains for 3 epochs on collected data
- Saves checkpoint every 100 steps
- Merges and saves final model to `models/gemma-finetuned-merged/`

**Hardware Requirements:**
- GPU: 8GB+ VRAM (RTX 4060 or better)
- RAM: 16GB+ system memory
- Time: ~1 hour for 100 examples, ~3-4 hours for 5000 examples

---

### Phase 4: Post-Fine-Tuning Evaluation (Measuring Improvement)

**Step 1: Load fine-tuned model**
```python
model, tokenizer = load_finetuned_model("models/gemma-finetuned-merged/")
```

**Step 2: Re-generate answers with improved model**
```python
# For each record that was evaluated
for record in pending_final_eval:
    # Generate NEW answer using fine-tuned model
    new_answer = model.generate(record["input"])
    
    # Compute metrics for new answer
    new_metrics = evaluate(new_answer)
    
    # Update database with tuned results
    UPDATE intune_db
    SET
      actual_output_tuned = new_answer,
      answer_relevancy_tuned = new_metrics.relevancy,
      contextual_precision_tuned = new_metrics.precision,
      ...
      status_eval_final = 'done'
```

**Step 3: Compare base vs fine-tuned**
The system now has both sets of metrics:

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|-----------|-------------|
| Answer Relevancy | 0.6500 | 0.7850 | +20.8% |
| Faithfulness | 0.5815 | 0.6067 | +4.3% |
| Overall Score | 0.2874 | 0.3073 | +6.9% |

**Step 4: Store results and continue**
- Improvements are visible in Frontend dashboard
- System continues collecting more data
- Process can repeat with new fine-tuning cycles

---

## 🔌 API Endpoints

### Backend API (Port 8000)

#### `POST /generate`
**Purpose:** Generate model response for user input  
**Called by:** Frontend chat interface  
**Process:**
1. Receive user prompt
2. Call Ollama API with Gemma model
3. Get generated response
4. Insert to Supabase with `status_eval_first='created'`
5. Return response to user

**Request:**
```json
{
  "prompt": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "model": "gemma-finetuned"
}
```

**Data Flow:**
```
Frontend → POST /generate → Backend
                              ↓
                         Ollama API
                              ↓
                         Generate response
                              ↓
                         Save to Supabase
                              ↓
                         Return to Frontend
```

---

#### `GET /health`
**Purpose:** Check API and database connectivity  
**Called by:** Monitoring systems, Frontend health checks  

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model": "gemma-finetuned",
  "ollama": "running"
}
```

---

#### `POST /finetune`
**Purpose:** Start fine-tuning worker (if not running)  
**Called by:** Frontend admin panel or manual trigger  

**Response:**
```json
{
  "message": "Fine-tuning worker started",
  "status": "running"
}
```

---

#### `GET /finetune/status`
**Purpose:** Check fine-tuning progress  
**Called by:** Frontend to display progress  

**Response:**
```json
{
  "status": "training",
  "records_collected": 2534,
  "target": 5000,
  "progress_percent": 50.68,
  "eta_minutes": 120
}
```

---

### Evaluation Worker Endpoints (Port 8001)

These endpoints control background evaluation workers:

#### `GET /status`
**Purpose:** Check worker status

**Response:**
```json
{
  "first_eval_worker": true,
  "final_eval_worker": true,
  "uptime_seconds": 3600
}
```

#### `POST /start-all-workers`
**Purpose:** Start both evaluation workers

#### `POST /stop-all-workers`
**Purpose:** Stop both evaluation workers

#### `GET /metrics/pending`
**Purpose:** Check pending evaluations

**Response:**
```json
{
  "pending_first_evaluations": 25,
  "pending_final_evaluations": 10
}
```

---

## 💾 Database Schema

### Table: `intune_db`

```sql
CREATE TABLE intune_db (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Input data
  input TEXT NOT NULL,                    -- User's question
  expected_output TEXT,                   -- Reference answer (from teacher)
  context JSONB,                          -- Background information
  
  -- Base model output
  actual_output TEXT,                     -- Generated answer (base model)
  
  -- Status tracking
  status_eval_first VARCHAR(20),          -- 'created' → 'done'
  status_eval_final VARCHAR(20),          -- NULL → 'done'
  
  -- Base model metrics (multiply by 10000)
  answer_relevancy INTEGER,
  contextual_precision INTEGER,
  contextual_recall INTEGER,
  contextual_relevancy INTEGER,
  faithfulness INTEGER,
  toxicity INTEGER,
  hallucination_rate INTEGER,
  overall INTEGER,
  
  -- Fine-tuned model output
  actual_output_tuned TEXT,               -- Generated answer (fine-tuned)
  
  -- Fine-tuned model metrics
  answer_relevancy_tuned INTEGER,
  contextual_precision_tuned INTEGER,
  contextual_recall_tuned INTEGER,
  contextual_relevancy_tuned INTEGER,
  faithfulness_tuned INTEGER,
  toxicity_tuned INTEGER,
  hallucination_rate_tuned INTEGER,
  overall_tuned INTEGER
);
```

### Status Flow

```
status_eval_first:  NULL → 'created' → 'done'
status_eval_final:  NULL → 'done'
```

**Status meanings:**
- `status_eval_first = 'created'`: Waiting for first evaluation
- `status_eval_first = 'done'`: Base metrics computed, ready for fine-tuning
- `status_eval_final = 'done'`: Fine-tuned metrics computed, complete

---

## 🚀 Setup Guide for Beginners

### Prerequisites

1. **Python 3.10+**
2. **NVIDIA GPU** with 8GB+ VRAM (for fine-tuning)
3. **Ollama** (local LLM server)
4. **Supabase account** (free tier works)
5. **Git** for cloning repositories

---

### Step 1: Set Up Backend

```bash
# Clone the repository
git clone https://github.com/Self-eval-llm/Intune-Backend.git
cd Intune-Backend

# Install Python dependencies
pip install -r requirements_finetune.txt

# Configure environment
cp config/.env.example .env
# Edit .env and add your Supabase credentials
```

---

### Step 2: Set Up Database

1. Go to [supabase.com](https://supabase.com) and create a free project
2. In Supabase SQL Editor, run these files in order:
   - `sql/supabase_setup.sql` - Creates main table
   - `sql/supabase_add_metrics.sql` - Adds metric columns
   - `sql/add_tuned_columns.sql` - Adds fine-tuned metric columns

3. Get your credentials:
   - Go to Settings → API
   - Copy `URL` and `anon/public key`
   - Add to `.env` file

---

### Step 3: Set Up Ollama

```bash
# Install Ollama (see ollama.ai)
# Start Ollama server
ollama serve

# Pull required models (in new terminal)
ollama pull gemma3:1b        # Base model for inference
ollama pull gpt-oss:20b      # Teacher model for data generation
```

---

### Step 4: Start the Backend

Open **3 terminals**:

**Terminal 1: API Server**
```bash
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```
- Handles `/generate` endpoint
- Serves responses to Frontend

**Terminal 2: First Evaluation Worker**
```bash
python app/eval_first.py
```
- Monitors for new records
- Computes base model metrics
- Updates status to 'done'

**Terminal 3: Fine-tune Worker**
```bash
python app/eval_finetune.py
```
- Monitors record count
- Triggers fine-tuning at threshold
- Evaluates fine-tuned model
- Updates final metrics

---

### Step 5: Set Up Frontend (If Available)

```bash
# Clone frontend repository
git clone https://github.com/Self-eval-llm/[Frontend-Repo].git
cd [Frontend-Repo]

# Install dependencies (React/Vue)
npm install

# Start development server
npm run dev
```

Frontend typically runs on `http://localhost:5173` or `http://localhost:3000`

---

### Step 6: Test the System

1. **Open Frontend** in browser: `http://localhost:5173`
2. **Type a question** in the chat: "What is AI?"
3. **Check Backend logs** - you'll see:
   ```
   INFO: Received prompt: What is AI?
   INFO: Generated response in 3.2s
   INFO: Inserted record to intune_db
   ```
4. **Check eval_first.py logs**:
   ```
   INFO: Found 1 records to evaluate
   INFO: Evaluating record 123
   INFO: ✓ Updated record 123
   ```
5. **Wait for fine-tuning** (when threshold reached):
   ```
   INFO: 🎯 Conditions met! Preparing training data...
   INFO: Starting fine-tuning process...
   INFO: ✅ Fine-tuning completed successfully
   INFO: Starting final evaluation with fine-tuned model...
   ```

---

## 🔁 The Self-Improvement Loop

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  1. USER ASKS QUESTION                                     │
│     ↓                                                      │
│  2. MODEL GENERATES ANSWER                                 │
│     ↓                                                      │
│  3. METRICS COMPUTED (How good is the answer?)             │
│     ↓                                                      │
│  4. SAVE TO DATABASE                                       │
│     ↓                                                      │
│  5. REPEAT 1-4 UNTIL THRESHOLD (e.g., 5000 examples)      │
│     ↓                                                      │
│  6. FINE-TUNE MODEL (Learn from examples)                  │
│     ↓                                                      │
│  7. RE-EVALUATE (Measure improvement)                      │
│     ↓                                                      │
│  8. COMPARE BASE vs FINE-TUNED                             │
│     ↓                                                      │
│  9. DEPLOY IMPROVED MODEL                                  │
│     ↓                                                      │
│  10. BACK TO STEP 1 (with better model)                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Why This Creates Continuous Improvement

1. **Data Collection**: Every user interaction creates a training example
2. **Quality Assessment**: Metrics identify weak areas automatically
3. **Targeted Training**: Fine-tuning focuses on improving identified weaknesses
4. **Measurable Results**: Before/after comparisons prove improvement
5. **Incremental Learning**: Each cycle builds on previous improvements
6. **No Manual Labeling**: System generates its own training data

---

## 🎓 Key Concepts Explained

### What is LoRA (Low-Rank Adaptation)?

**Traditional Fine-tuning:**
- Modifies ALL model parameters (billions of numbers)
- Requires massive GPU memory (40GB+)
- Takes days to train

**LoRA Fine-tuning:**
- Adds small "adapter" layers (~1% of model size)
- Only trains the adapters, freezes base model
- Requires only 8GB GPU memory
- Trains in hours instead of days
- Can merge adapters back into model later

**Analogy:** Instead of rewriting an entire book, you add sticky notes with corrections.

---

### What are Evaluation Metrics?

Think of metrics as a report card for the model:

- **Answer Relevancy**: Does the answer actually address the question?
  - Bad: Q: "What is AI?" A: "The weather is nice today."
  - Good: Q: "What is AI?" A: "AI is artificial intelligence..."

- **Faithfulness**: Does the answer stick to the provided facts?
  - Bad: Making up information not in the context
  - Good: Only stating what's in the reference material

- **Hallucination Rate**: How much is made up?
  - Low (good): Model admits "I don't know" when uncertain
  - High (bad): Model confidently states false information

---

### Why Separate Workers?

**Design Decision:** Three separate processes instead of one monolithic application

**Benefits:**
1. **Resilience**: If one crashes, others continue
2. **Scalability**: Can run on different machines
3. **Resource Management**: GPU-intensive fine-tuning doesn't block API
4. **Debugging**: Easier to trace issues to specific components
5. **Development**: Teams can work on different parts independently

---

### Why INT8 for Metrics?

**Question:** Why store 0.7532 as 7532 instead of 0.7532?

**Answer:**
- PostgreSQL DECIMAL/FLOAT operations are slower than INTEGER
- INT8 takes 8 bytes, DECIMAL takes 16 bytes (50% storage savings)
- Integer comparisons are faster in database queries
- No loss of precision for 4 decimal places

**Conversion:**
```python
# Store: 0.7532 → 7532
stored_value = int(round(0.7532 * 10000))

# Retrieve: 7532 → 0.7532
actual_value = stored_value / 10000
```

---

## 🛠️ Advanced Usage

### Generating Training Data from Scratch

If you want to bootstrap without user interactions:

```bash
# Generate 5000 training examples using teacher model
python src/data_generation/teacher.py --n 5000 --mode continuous

# Generate base model outputs
python src/data_generation/student.py

# Compute metrics for all
python src/evaluation/update_metrics.py

# Prepare for training
python src/data_generation/prepare_data.py

# Fine-tune
python src/training/finetune.py

# Evaluate fine-tuned model
python src/evaluation/evaluate_finetuned.py
```

**Use case:** Batch training on curated datasets before deployment

---

### Monitoring and Observability

**Check system health:**
```bash
# API health
curl http://localhost:8000/health

# Worker status
curl http://localhost:8001/status

# Pending evaluations
curl http://localhost:8001/metrics/pending
```

**Database queries:**
```sql
-- Count records by status
SELECT 
  status_eval_first, 
  status_eval_final, 
  COUNT(*) 
FROM intune_db 
GROUP BY status_eval_first, status_eval_final;

-- Average metrics comparison
SELECT
  AVG(overall::float / 10000) AS base_avg,
  AVG(overall_tuned::float / 10000) AS tuned_avg,
  AVG(overall_tuned::float - overall::float) / 10000 AS improvement
FROM intune_db
WHERE status_eval_final = 'done';
```

---

### Customizing Fine-Tuning Parameters

Edit `src/training/finetune.py`:

```python
# For smaller GPU (4GB)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1024

# For larger GPU (24GB)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
MAX_SEQ_LENGTH = 4096

# Learning rate tuning
LEARNING_RATE = 2e-4  # Default
# Increase to 3e-4 for faster convergence
# Decrease to 1e-4 for more stable training

# Training duration
NUM_EPOCHS = 3  # Default
# Increase for more training (may overfit)
# Decrease for faster experimentation
```

---

## 📈 Performance Benchmarks

### API Response Times

| Operation | Average | 95th Percentile |
|-----------|---------|-----------------|
| `/generate` (simple) | 2.1s | 3.5s |
| `/generate` (complex) | 4.8s | 7.2s |
| Metric computation | 0.3s | 0.5s |
| Database write | 0.05s | 0.1s |

### Throughput

| Component | Records/Hour |
|-----------|--------------|
| API generation | 720 |
| First evaluation | 1200 |
| Fine-tuned generation | 180 |
| Final evaluation | 180 |

### Resource Usage

| Process | CPU | RAM | VRAM |
|---------|-----|-----|------|
| API server | 15% | 500MB | - |
| eval_first | 25% | 1GB | - |
| eval_finetune (idle) | 5% | 300MB | - |
| eval_finetune (training) | 80% | 8GB | 7.5GB |
| Ollama (Gemma 1B) | 30% | 2GB | 2GB |

---

## 🔧 Troubleshooting

### Issue: API returns 503 "Ollama service not running"

**Solution:**
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

---

### Issue: Workers not processing records

**Check 1: Database connectivity**
```python
python -c "from src.database.supabase_client import get_supabase_client; print(get_supabase_client())"
```

**Check 2: Status flags**
```sql
SELECT status_eval_first, COUNT(*) FROM intune_db GROUP BY status_eval_first;
```

**Check 3: Worker logs**
Look for errors in terminal output

---

### Issue: Fine-tuning fails with CUDA Out of Memory

**Solution 1: Reduce batch size**
```python
# In src/training/finetune.py
BATCH_SIZE = 1  # Down from 2
```

**Solution 2: Reduce sequence length**
```python
MAX_SEQ_LENGTH = 1024  # Down from 2048
```

**Solution 3: Use CPU (slower but works)**
```python
load_in_4bit = False
# Add: device_map = "cpu"
```

---

### Issue: Metrics show no improvement after fine-tuning

**Possible causes:**
1. **Insufficient training data**: Need at least 1000+ examples
2. **Poor quality data**: Garbage in, garbage out
3. **Overfitting**: Training too many epochs on small dataset
4. **Wrong learning rate**: Too high (unstable) or too low (no learning)

**Solutions:**
- Collect more diverse training data
- Review `data/raw/training_dataset.json` for quality
- Reduce `NUM_EPOCHS` from 3 to 1 for small datasets
- Try different `LEARNING_RATE` values (1e-4 to 3e-4)

---

## 🎨 Frontend Integration Guide

### Required API Calls

**1. Generate response:**
```javascript
async function generateResponse(prompt) {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  });
  return await response.json();
}
```

**2. Check fine-tuning status:**
```javascript
async function getFineTuningStatus() {
  const response = await fetch('http://localhost:8000/finetune/status');
  return await response.json();
}
```

**3. Get pending evaluations:**
```javascript
async function getPendingCount() {
  const response = await fetch('http://localhost:8001/metrics/pending');
  return await response.json();
}
```

### Example Chat Component (React)

```jsx
import React, { useState } from 'react';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    setLoading(true);
    
    // Add user message
    setMessages([...messages, { role: 'user', content: input }]);
    
    // Call backend API
    const response = await fetch('http://localhost:8000/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input })
    });
    
    const data = await response.json();
    
    // Add assistant message
    setMessages([
      ...messages,
      { role: 'user', content: input },
      { role: 'assistant', content: data.response }
    ]);
    
    setInput('');
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Generating...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

---

## 📚 Additional Resources

### Repository Structure
```
Intune-Backend/
├── app/                      # API and workers
│   ├── app.py               # Main API server
│   ├── eval_first.py        # First evaluation worker
│   └── eval_finetune.py     # Fine-tuning worker
├── src/                      # Core logic
│   ├── data_generation/     # Data pipeline
│   ├── training/            # Fine-tuning
│   ├── evaluation/          # Metrics computation
│   ├── metrics/             # Evaluation engine
│   └── database/            # Supabase client
├── models/                   # Model storage
├── data/                     # Training data
├── sql/                      # Database schemas
└── config/                   # Configuration
```

### Related Documentation
- [README.md](README.md) - Quick start guide
- [RUNNING.md](RUNNING.md) - How to run the system
- [app/README.md](app/README.md) - API documentation
- [requirements_finetune.txt](requirements_finetune.txt) - Python dependencies

### External Links
- [Ollama Documentation](https://ollama.ai/docs)
- [Supabase Documentation](https://supabase.com/docs)
- [Unsloth (LoRA Training)](https://github.com/unslothai/unsloth)
- [Gemma Model](https://ai.google.dev/gemma)

---

## 🎯 Summary

This self-improving LLM framework demonstrates how to create a **continuous learning system** that:

1. ✅ **Collects data automatically** from user interactions
2. ✅ **Evaluates quality objectively** with 8 metrics
3. ✅ **Improves autonomously** through fine-tuning
4. ✅ **Measures progress quantitatively** with before/after comparisons
5. ✅ **Scales efficiently** using LoRA and batch processing
6. ✅ **Operates continuously** with background workers

**Three interconnected components:**
- 🔷 **Backend**: Processing engine (Python, FastAPI, PyTorch)
- 🔷 **Frontend**: User interface (React/Vue)
- 🔷 **Database**: Persistent storage (Supabase/PostgreSQL)

**For beginners:** Start with the setup guide, run through a test cycle with 10-20 examples, observe the workflow, then scale up.

**For advanced users:** Customize metrics, adjust training parameters, integrate into production systems, implement continuous deployment.

---

**Questions or issues?** Check the troubleshooting section or review the code comments for detailed explanations.

**Contributing?** Ensure you understand the data flow and status transitions before making changes.

**Ready to start?** Jump to the [Setup Guide](#-setup-guide-for-beginners)! 🚀
