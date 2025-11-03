-- SQL script to add metric columns to the inference_results table
-- Run this in your Supabase SQL Editor BEFORE running process_and_upload.py

-- Drop columns if they exist with wrong type, then recreate
ALTER TABLE inference_results
DROP COLUMN IF EXISTS answer_relevancy,
DROP COLUMN IF EXISTS contextual_precision,
DROP COLUMN IF EXISTS contextual_recall,
DROP COLUMN IF EXISTS contextual_relevancy,
DROP COLUMN IF EXISTS faithfulness,
DROP COLUMN IF EXISTS toxicity,
DROP COLUMN IF EXISTS hallucination_rate,
DROP COLUMN IF EXISTS overall;

-- Add metric columns (using INT8/BIGINT type)
-- Stores values multiplied by 10000 to preserve 4 decimal places
-- Example: 0.8534 is stored as 8534

ALTER TABLE inference_results
ADD COLUMN answer_relevancy INT8,
ADD COLUMN contextual_precision INT8,
ADD COLUMN contextual_recall INT8,
ADD COLUMN contextual_relevancy INT8,
ADD COLUMN faithfulness INT8,
ADD COLUMN toxicity INT8,
ADD COLUMN hallucination_rate INT8,
ADD COLUMN overall INT8;

-- Create indexes on key metrics for faster queries
CREATE INDEX IF NOT EXISTS idx_inference_results_overall ON inference_results(overall DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_faithfulness ON inference_results(faithfulness DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_toxicity ON inference_results(toxicity ASC);

-- Verify the changes
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'inference_results'
AND column_name IN (
    'answer_relevancy',
    'contextual_precision',
    'contextual_recall',
    'contextual_relevancy',
    'faithfulness',
    'toxicity',
    'hallucination_rate',
    'overall'
)
ORDER BY column_name;
