-- ============================================================================
-- Step 5: Add Batch Finetuning Columns
-- ============================================================================
-- Run this in Supabase SQL Editor BEFORE running batch finetune notebook
-- These columns store the output from finetuning on ALL 50K at once

-- Batch finetuned model output (trained on full 50K in one go)
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_batch TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_batch DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_batch DECIMAL(5,4);

-- Index for batch generation progress tracking
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_batch_pending 
    ON modelcomp_50k(student_output_batch) 
    WHERE student_output_batch IS NULL;

-- Updated comparison view: incremental stages + batch
CREATE OR REPLACE VIEW incremental_vs_batch AS
SELECT 
    'Base Student' as approach,
    COUNT(student_output) FILTER (WHERE student_output IS NOT NULL) as outputs_done,
    NULL::numeric as avg_score,
    ROUND(AVG(generation_latency)::numeric, 3) as avg_latency
FROM modelcomp_50k
UNION ALL
SELECT 
    'Stage 1 (5K incr)', 
    COUNT(student_output_ckpt1) FILTER (WHERE student_output_ckpt1 IS NOT NULL),
    ROUND(AVG(score_ckpt1)::numeric, 4),
    ROUND(AVG(latency_ckpt1)::numeric, 3)
FROM modelcomp_50k
UNION ALL
SELECT 'Stage 2 (10K incr)', COUNT(student_output_ckpt2) FILTER (WHERE student_output_ckpt2 IS NOT NULL), ROUND(AVG(score_ckpt2)::numeric, 4), ROUND(AVG(latency_ckpt2)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 3 (15K incr)', COUNT(student_output_ckpt3) FILTER (WHERE student_output_ckpt3 IS NOT NULL), ROUND(AVG(score_ckpt3)::numeric, 4), ROUND(AVG(latency_ckpt3)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 4 (20K incr)', COUNT(student_output_ckpt4) FILTER (WHERE student_output_ckpt4 IS NOT NULL), ROUND(AVG(score_ckpt4)::numeric, 4), ROUND(AVG(latency_ckpt4)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 5 (25K incr)', COUNT(student_output_ckpt5) FILTER (WHERE student_output_ckpt5 IS NOT NULL), ROUND(AVG(score_ckpt5)::numeric, 4), ROUND(AVG(latency_ckpt5)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 6 (30K incr)', COUNT(student_output_ckpt6) FILTER (WHERE student_output_ckpt6 IS NOT NULL), ROUND(AVG(score_ckpt6)::numeric, 4), ROUND(AVG(latency_ckpt6)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 7 (35K incr)', COUNT(student_output_ckpt7) FILTER (WHERE student_output_ckpt7 IS NOT NULL), ROUND(AVG(score_ckpt7)::numeric, 4), ROUND(AVG(latency_ckpt7)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 8 (40K incr)', COUNT(student_output_ckpt8) FILTER (WHERE student_output_ckpt8 IS NOT NULL), ROUND(AVG(score_ckpt8)::numeric, 4), ROUND(AVG(latency_ckpt8)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 9 (45K incr)', COUNT(student_output_ckpt9) FILTER (WHERE student_output_ckpt9 IS NOT NULL), ROUND(AVG(score_ckpt9)::numeric, 4), ROUND(AVG(latency_ckpt9)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 10 (50K incr)', COUNT(student_output_ckpt10) FILTER (WHERE student_output_ckpt10 IS NOT NULL), ROUND(AVG(score_ckpt10)::numeric, 4), ROUND(AVG(latency_ckpt10)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 
    'Batch (50K at once)', 
    COUNT(student_output_batch) FILTER (WHERE student_output_batch IS NOT NULL),
    ROUND(AVG(score_batch)::numeric, 4),
    ROUND(AVG(latency_batch)::numeric, 3)
FROM modelcomp_50k;
