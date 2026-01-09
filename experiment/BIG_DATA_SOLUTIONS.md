# Big Data Analytics Solutions for LLM Distillation Pipeline

## Overview

This document outlines **data-intensive computing solutions** for the LLM distillation project. These solutions run **independently** of the teacher selection experiment (4K/50K) and focus on **real-time streaming analytics** and **batch processing** for LLM inference metrics.

---

## Architecture Overview

```
┌─────────┐     ┌─────────────┐     ┌─────────────────────┐
│  User   │────▶│  FastAPI    │────▶│  Ollama Server      │
└─────────┘     │  App        │◀────│  (Gemma 1B)         │
                └──────┬──────┘     └─────────────────────┘
                       │
                       │ send event
                       ▼
                ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
                │   Kafka     │────▶│  Spark Streaming    │────▶│  Supabase   │
                │ llm_inputs  │     │  (Analytics)        │     │  or HDFS    │
                └─────────────┘     └─────────────────────┘     └─────────────┘
```

---

## Problem Statements for Big Data Solutions

### Problem 1: Real-Time LLM Inference Monitoring

**Data-Intensive Argument:**
- High volume of inference requests (1000+ requests/minute at scale)
- Need real-time latency tracking, token throughput, and error rates
- Streaming aggregations (windowed averages, percentiles)

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Queue | Apache Kafka | Buffer inference events |
| Stream Processor | Spark Structured Streaming | Real-time aggregations |
| Storage | Supabase / TimescaleDB | Time-series metrics |
| Dashboard | Grafana | Live monitoring |

**Kafka Topics:**
```
llm_inputs        - Raw prompts with metadata
llm_outputs       - Generated responses with timing
llm_metrics       - Computed quality scores
llm_errors        - Failed requests for retry
```

---

### Problem 2: Batch Quality Evaluation at Scale

**Data-Intensive Argument:**
- 50,000+ records need evaluation metrics computed
- CPU-bound text processing (tokenization, cosine similarity)
- Embarrassingly parallel workload

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Distributed Compute | Apache Spark | Parallel metric computation |
| Data Lake | HDFS / MinIO | Store raw outputs |
| Orchestration | Apache Airflow | Schedule batch jobs |
| Results | Supabase | Query-optimized storage |

**Spark Job Design:**
```python
# Pseudo-code for distributed evaluation
df = spark.read.parquet("hdfs://outputs/")

# Parallel UDF for metrics
@udf(returnType=StructType([...]))
def compute_metrics(instruction, output, context):
    return {
        "task_success": score_task_success(...),
        "coverage": score_coverage(...),
        "hallucination": score_hallucination(...),
    }

results = df.withColumn("metrics", compute_metrics(...))
results.write.jdbc(supabase_url, "evaluation_results")
```

---

### Problem 3: Distributed Inference Load Balancing

**Data-Intensive Argument:**
- Multiple Ollama instances across machines (Windows + MacBook)
- Need to balance load based on GPU utilization
- Handle machine failures gracefully

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Load Balancer | NGINX / HAProxy | Route requests |
| Service Registry | Consul / etcd | Track healthy nodes |
| Queue | Redis / Kafka | Request buffering |
| Monitoring | Prometheus | Resource metrics |

**Architecture:**
```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Ollama Node │   │ Ollama Node │   │ Ollama Node │
    │ (Windows)   │   │ (MacBook)   │   │ (Cloud GPU) │
    └─────────────┘   └─────────────┘   └─────────────┘
```

---

### Problem 4: Prompt-Response Data Lake

**Data-Intensive Argument:**
- Store all prompts, responses, and metadata for analysis
- Enable ad-hoc queries on historical data
- Support schema evolution as metrics change

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Storage Format | Parquet / Delta Lake | Columnar, versioned |
| Query Engine | Apache Spark SQL / DuckDB | Ad-hoc analytics |
| Catalog | Hive Metastore | Schema management |
| Versioning | Delta Lake / Iceberg | Time travel queries |

**Data Schema:**
```
llm_data/
├── year=2026/
│   ├── month=01/
│   │   ├── day=09/
│   │   │   ├── part-00000.parquet
│   │   │   ├── part-00001.parquet
│   │   │   └── ...
```

**Partitioning Strategy:**
- By date (for time-based queries)
- By task_label (for category analysis)
- By model_version (for A/B comparison)

---

### Problem 5: Streaming Anomaly Detection

**Data-Intensive Argument:**
- Detect quality degradation in real-time
- Alert when hallucination rate spikes
- Identify distribution shift in prompts

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Stream Processing | Kafka Streams / Flink | Stateful processing |
| ML | Spark MLlib / River | Online learning |
| Alerting | PagerDuty / Slack | Notifications |

**Metrics to Monitor:**
```
- Rolling average hallucination rate (5-min window)
- P95 latency per model
- Token throughput per minute
- Error rate by error type
- Prompt length distribution shift
```

---

### Problem 6: A/B Testing Infrastructure

**Data-Intensive Argument:**
- Route traffic between teacher models (Alpaca vs OSS)
- Collect paired samples for statistical comparison
- Real-time significance testing

**Solution Components:**
| Component | Technology | Purpose |
|-----------|------------|---------|
| Traffic Splitting | Feature flags (LaunchDarkly/custom) | Route requests |
| Event Collection | Kafka | Capture all variants |
| Analysis | Spark + SciPy | Statistical tests |
| Reporting | Jupyter / Superset | Visualizations |

**Experiment Design:**
```
Request → Hash(user_id) → 
    if hash % 2 == 0: use tuned_alpaca
    else: use tuned_oss20b
    
Log: {request_id, variant, metrics, timestamp}
```

---

## Implementation Priority

| Priority | Problem | Complexity | Value |
|----------|---------|------------|-------|
| 🔴 P0 | Real-Time Monitoring | Medium | High |
| 🔴 P0 | Batch Evaluation at Scale | Low | High |
| 🟡 P1 | Data Lake Setup | Medium | Medium |
| 🟡 P1 | A/B Testing | Medium | High |
| 🟢 P2 | Distributed Inference | High | Medium |
| 🟢 P2 | Anomaly Detection | High | Medium |

---

## Recommended Stack

### Minimal Setup (Local Development)
```
FastAPI → Kafka (Docker) → Spark Local → Supabase
```

### Production Setup
```
FastAPI (Kubernetes) → Kafka (Confluent Cloud) → Spark (Databricks/EMR) → 
    → Supabase + S3/HDFS
```

---

## Data Volumes Estimation

| Dataset | Records | Size (Est.) | Processing Time |
|---------|---------|-------------|-----------------|
| 4K Teacher Comparison | 4,000 | ~50 MB | 5-10 min |
| 50K Full Dataset | 50,000 | ~500 MB | 1-2 hours |
| Daily Inference Logs | 100,000 | ~1 GB | Streaming |
| Historical Archive | 10M+ | ~100 GB | Batch (nightly) |

---

## Kafka Topic Design

### Topic: `llm_inputs`
```json
{
  "request_id": "uuid",
  "timestamp": "2026-01-09T10:30:00Z",
  "prompt": "...",
  "context": ["..."],
  "model": "gemma-1b",
  "user_id": "hash",
  "metadata": {
    "label": "technical_code",
    "source": "api"
  }
}
```

### Topic: `llm_outputs`
```json
{
  "request_id": "uuid",
  "timestamp": "2026-01-09T10:30:02Z",
  "response": "...",
  "tokens_generated": 128,
  "latency_ms": 2340,
  "model": "gemma-1b"
}
```

### Topic: `llm_metrics`
```json
{
  "request_id": "uuid",
  "timestamp": "2026-01-09T10:30:03Z",
  "metrics": {
    "task_success": 0.85,
    "hallucination": 0.12,
    "faithfulness": 0.78,
    "overall": 0.76
  }
}
```

---

## Spark Streaming Job Template

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("LLM-Metrics-Streaming") \
    .getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "llm_outputs") \
    .load()

# Parse JSON
schema = StructType([
    StructField("request_id", StringType()),
    StructField("latency_ms", IntegerType()),
    StructField("tokens_generated", IntegerType()),
])

parsed = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Windowed aggregations
windowed = parsed \
    .withWatermark("timestamp", "1 minute") \
    .groupBy(window("timestamp", "5 minutes")) \
    .agg(
        avg("latency_ms").alias("avg_latency"),
        percentile_approx("latency_ms", 0.95).alias("p95_latency"),
        sum("tokens_generated").alias("total_tokens"),
        count("*").alias("request_count")
    )

# Write to Supabase/console
query = windowed.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()
```

---

## Next Steps

1. **[ ] Set up Kafka locally** (Docker Compose)
2. **[ ] Modify FastAPI to publish events**
3. **[ ] Create Spark streaming job**
4. **[ ] Set up Supabase tables for metrics**
5. **[ ] Build Grafana dashboard**
6. **[ ] Implement batch evaluation job**

---

## Files to Create

| File | Purpose |
|------|---------|
| `docker-compose.kafka.yml` | Kafka + Zookeeper setup |
| `app/kafka_producer.py` | Publish events from FastAPI |
| `spark/streaming_metrics.py` | Real-time aggregations |
| `spark/batch_evaluation.py` | Distributed evaluation |
| `sql/metrics_tables.sql` | Supabase schema |

---

## References

- Apache Kafka Documentation
- Spark Structured Streaming Guide
- Delta Lake for Data Lakes
- Supabase Real-time Subscriptions
