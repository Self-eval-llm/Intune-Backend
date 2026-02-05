-- Add columns for incremental learning outputs (10 stages)
-- Each stage stores the student model output after training on cumulative data

-- Student outputs for each checkpoint/stage
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt1 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt2 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt3 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt4 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt5 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt6 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt7 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt8 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt9 TEXT;
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS student_output_ckpt10 TEXT;

-- Overall scores for each checkpoint (to track improvement)
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt1 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt2 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt3 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt4 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt5 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt6 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt7 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt8 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt9 DECIMAL(5,4);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS score_ckpt10 DECIMAL(5,4);

-- Latency per checkpoint (to track if model gets slower/faster)
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt1 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt2 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt3 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt4 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt5 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt6 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt7 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt8 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt9 DECIMAL(8,3);
ALTER TABLE modelcomp_50k ADD COLUMN IF NOT EXISTS latency_ckpt10 DECIMAL(8,3);

-- Create index on checkpoint for faster stage-based queries
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_checkpoint ON modelcomp_50k(checkpoint);

-- View to compare scores across checkpoints
CREATE OR REPLACE VIEW incremental_progress AS
SELECT 
    checkpoint,
    COUNT(*) as record_count,
    AVG(score_ckpt1) as avg_score_ckpt1,
    AVG(score_ckpt2) as avg_score_ckpt2,
    AVG(score_ckpt3) as avg_score_ckpt3,
    AVG(score_ckpt4) as avg_score_ckpt4,
    AVG(score_ckpt5) as avg_score_ckpt5,
    AVG(score_ckpt6) as avg_score_ckpt6,
    AVG(score_ckpt7) as avg_score_ckpt7,
    AVG(score_ckpt8) as avg_score_ckpt8,
    AVG(score_ckpt9) as avg_score_ckpt9,
    AVG(score_ckpt10) as avg_score_ckpt10
FROM modelcomp_50k
GROUP BY checkpoint
ORDER BY checkpoint;

-- Summary view for quick progress check
CREATE OR REPLACE VIEW stage_summary AS
SELECT
    'Stage 1 (5K)' as stage,
    COUNT(student_output_ckpt1) FILTER (WHERE student_output_ckpt1 IS NOT NULL) as outputs_generated,
    ROUND(AVG(score_ckpt1)::numeric, 4) as avg_score,
    ROUND(AVG(latency_ckpt1)::numeric, 3) as avg_latency_ms
FROM modelcomp_50k
UNION ALL
SELECT 'Stage 2 (10K)', COUNT(student_output_ckpt2) FILTER (WHERE student_output_ckpt2 IS NOT NULL), ROUND(AVG(score_ckpt2)::numeric, 4), ROUND(AVG(latency_ckpt2)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 3 (15K)', COUNT(student_output_ckpt3) FILTER (WHERE student_output_ckpt3 IS NOT NULL), ROUND(AVG(score_ckpt3)::numeric, 4), ROUND(AVG(latency_ckpt3)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 4 (20K)', COUNT(student_output_ckpt4) FILTER (WHERE student_output_ckpt4 IS NOT NULL), ROUND(AVG(score_ckpt4)::numeric, 4), ROUND(AVG(latency_ckpt4)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 5 (25K)', COUNT(student_output_ckpt5) FILTER (WHERE student_output_ckpt5 IS NOT NULL), ROUND(AVG(score_ckpt5)::numeric, 4), ROUND(AVG(latency_ckpt5)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 6 (30K)', COUNT(student_output_ckpt6) FILTER (WHERE student_output_ckpt6 IS NOT NULL), ROUND(AVG(score_ckpt6)::numeric, 4), ROUND(AVG(latency_ckpt6)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 7 (35K)', COUNT(student_output_ckpt7) FILTER (WHERE student_output_ckpt7 IS NOT NULL), ROUND(AVG(score_ckpt7)::numeric, 4), ROUND(AVG(latency_ckpt7)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 8 (40K)', COUNT(student_output_ckpt8) FILTER (WHERE student_output_ckpt8 IS NOT NULL), ROUND(AVG(score_ckpt8)::numeric, 4), ROUND(AVG(latency_ckpt8)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 9 (45K)', COUNT(student_output_ckpt9) FILTER (WHERE student_output_ckpt9 IS NOT NULL), ROUND(AVG(score_ckpt9)::numeric, 4), ROUND(AVG(latency_ckpt9)::numeric, 3) FROM modelcomp_50k
UNION ALL
SELECT 'Stage 10 (50K)', COUNT(student_output_ckpt10) FILTER (WHERE student_output_ckpt10 IS NOT NULL), ROUND(AVG(score_ckpt10)::numeric, 4), ROUND(AVG(latency_ckpt10)::numeric, 3) FROM modelcomp_50k;
