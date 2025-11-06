-- Add columns for prompt-tuned outputs and metrics
-- Run this in Supabase SQL Editor before running prompt_tuning.py

ALTER TABLE inference_results
ADD COLUMN IF NOT EXISTS actual_output_tuned TEXT,
ADD COLUMN IF NOT EXISTS answer_relevancy_tuned INT8,
ADD COLUMN IF NOT EXISTS contextual_precision_tuned INT8,
ADD COLUMN IF NOT EXISTS contextual_recall_tuned INT8,
ADD COLUMN IF NOT EXISTS contextual_relevancy_tuned INT8,
ADD COLUMN IF NOT EXISTS faithfulness_tuned INT8,
ADD COLUMN IF NOT EXISTS toxicity_tuned INT8,
ADD COLUMN IF NOT EXISTS hallucination_rate_tuned INT8,
ADD COLUMN IF NOT EXISTS overall_tuned INT8;

-- Create indexes for tuned metrics
CREATE INDEX IF NOT EXISTS idx_inference_results_overall_tuned ON inference_results(overall_tuned DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_faithfulness_tuned ON inference_results(faithfulness_tuned DESC);

-- Verify the columns were added
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'inference_results'
AND column_name LIKE '%tuned%'
ORDER BY column_name;
