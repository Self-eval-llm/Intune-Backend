-- Add columns for fine-tuned model outputs and metrics
-- Run this in Supabase SQL Editor

-- Output columns
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS tuned_alpaca TEXT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS tuned_oss20b TEXT;

-- Alpaca model metrics (INT8 format: value * 10000)
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_answer_relevancy INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_contextual_precision INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_contextual_recall INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_contextual_relevancy INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_faithfulness INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_toxicity INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_hallucination_rate INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_overall INT;

-- OSS-20B model metrics (INT8 format: value * 10000)
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_answer_relevancy INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_contextual_precision INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_contextual_recall INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_contextual_relevancy INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_faithfulness INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_toxicity INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_hallucination_rate INT;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss20b_overall INT;

-- Verify columns were added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'modelComp' 
ORDER BY ordinal_position;
