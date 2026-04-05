# INTUNE Backend

An event-driven backend for LLM inference, evaluation, and fine-tuning with Supabase, Kafka, and Python workers.

## Team Members

- Abhang Pawar - 24bds054
- Radhakrishna Bharuka - 24bds063
- Nilesh Dwivedi - 24bds048
- Rushikesh Masalkar - 24bds040
- Ashish Dargupally - 24bds015

## Repository Structure

```text
Intune-Backend/
|-- app/                         # FastAPI app and evaluation workers
|-- config/                      # Project configuration files
|-- data/                        # Raw, processed, and experiment datasets
|-- docs/                        # Guides, diagrams, and media assets
|-- event_driven_pipeline/       # Realtime bridge, trigger logic, validators, tests
|-- experiment/                  # Research pipeline scripts for incremental vs batch learning
|-- models/                      # Finetuned and merged model artifacts
|-- reports/                     # Evaluation and comparison reports
|-- scripts/                     # Utility scripts for model merge/convert/reporting
|-- sql/                         # Database schema scripts and migration SQL
|-- src/                         # Core source modules (database, metrics, training, evaluation)
|-- commands.txt                 # End-to-end command reference
|-- requirements.txt             # Python dependencies
|-- run_spark.ps1                # PowerShell helper to run Spark trigger job
|-- intune_REVISED.tex           # Project report/paper source
```

## Prerequisites

- Python 3.9.x
- pip
- Ollama (for local model serving)
- Kafka running locally (or reachable remotely)
- Supabase project and credentials

## Installation and Implementation Process

### 1) Set up Python environment

```bash
python3 -m venv intune_pipeline_env
source intune_pipeline_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2) Configure environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_or_service_key
SUPABASE_SERVICE_ROLE_KEY=optional_service_role_key

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_EVENTS=intune.status.events
KAFKA_TOPIC_TRIGGERS=pipeline.triggers
KAFKA_TOPIC_DLQ=pipeline.dlq
KAFKA_GROUP_ID=pipeline-trigger-consumer

REALTIME_TABLE=intune_db
REALTIME_SCHEMA=public

TRIGGER_THRESHOLD=2
LOG_LEVEL=INFO
```

### 3) Apply database schema

Run SQL scripts in order from the `sql/` folder:

1. `01_schema_setup.sql`
2. `02_schema_eval_matrix.sql`
3. `03_schema_incremental_tables.sql`
4. `04_schema_50k_checkpoints.sql`
5. `05_schema_batch_columns.sql`
6. `05_schema_incremental_pipeline.sql`
7. `06_cleanup_legacy_columns.sql`
8. `07_schema_distributed_worker.sql`

### 4) Implementation flow in this codebase

The repository supports two practical execution modes:

- API + worker mode (FastAPI inference and evaluation workers)
- Event-driven pipeline mode (Supabase Realtime -> Kafka -> Trigger consumer)

## Executing the Code

### A) Run the FastAPI inference app

```bash
source intune_pipeline_env/bin/activate
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/
```

Generate response:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain knowledge distillation in simple terms."}'
```

Optional worker execution:

```bash
python app/eval_first.py
python app/eval_finetune.py
```

### B) Run the event-driven pipeline

Use separate terminals after activating the virtual environment in each terminal.

Terminal 1:

```bash
python event_driven_pipeline/realtime_kafka_bridge.py
```

Terminal 2:

```bash
python event_driven_pipeline/spark_pipeline_trigger_job_standalone.py
```

Terminal 3:

```bash
python event_driven_pipeline/trigger_consumer.py
```

Terminal 4 (test data generator):

```bash
python event_driven_pipeline/manual_test_data_generator.py
```

### C) Run research pipeline scripts

```bash
python experiment/pipeline.py --status
python experiment/pipeline.py --context
python experiment/pipeline.py --generate --checkpoint 1
python experiment/pipeline.py --mode incremental --checkpoint 1 --run-all
python experiment/pipeline.py --mode batch --run-all
```

## 2-Minute Demo of the Working App

Demo videos:

- [2-Minute App Demo - Main Video](bda_video.mp4)
- [2-Minute App Demo - Docs Copy](docs/media/demo_video.mp4)
