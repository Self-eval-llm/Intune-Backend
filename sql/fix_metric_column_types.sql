-- ============================================================================
-- FIX: Convert all metric columns to INT type
-- ============================================================================
-- This script fixes the "numeric field overflow" error
-- by converting NUMERIC(5,4) columns to INT
--
-- The error occurred because:
-- - NUMERIC(5,4) = 5 total digits, 4 decimal places
-- - Valid range: -9.9999 to 9.9999
-- - Our metrics: 0-10000 (from to_int8() multiplying by 10000)
--
-- Solution: Use INT columns instead
-- ============================================================================

-- Check current column types
SELECT column_name, data_type, numeric_precision, numeric_scale
FROM information_schema.columns
WHERE table_name = 'modelComp'
AND column_name LIKE '%alpaca_%' OR column_name LIKE '%oss_%'
ORDER BY column_name;

-- ============================================================================
-- PART 1: Alpaca Metrics - Convert to INT
-- ============================================================================

-- Drop and recreate alpaca metric columns as INT
ALTER TABLE "modelComp"
  DROP COLUMN IF EXISTS alpaca_struct_correct,
  DROP COLUMN IF EXISTS alpaca_task_success,
  DROP COLUMN IF EXISTS alpaca_instr_follow,
  DROP COLUMN IF EXISTS alpaca_coverage,
  DROP COLUMN IF EXISTS alpaca_faithfulness,
  DROP COLUMN IF EXISTS alpaca_hallucination,
  DROP COLUMN IF EXISTS alpaca_ctx_grounding,
  DROP COLUMN IF EXISTS alpaca_overall;

ALTER TABLE "modelComp"
  ADD COLUMN alpaca_struct_correct INT DEFAULT 0,
  ADD COLUMN alpaca_task_success INT DEFAULT 0,
  ADD COLUMN alpaca_instr_follow INT DEFAULT 0,
  ADD COLUMN alpaca_coverage INT DEFAULT 0,
  ADD COLUMN alpaca_faithfulness INT DEFAULT 0,
  ADD COLUMN alpaca_hallucination INT DEFAULT 0,
  ADD COLUMN alpaca_ctx_grounding INT DEFAULT 0,
  ADD COLUMN alpaca_overall INT DEFAULT 0;

-- ============================================================================
-- PART 2: OSS-20B Metrics - Convert to INT
-- ============================================================================

-- Drop and recreate OSS metric columns as INT
ALTER TABLE "modelComp"
  DROP COLUMN IF EXISTS oss_struct_correct,
  DROP COLUMN IF EXISTS oss_task_success,
  DROP COLUMN IF EXISTS oss_instr_follow,
  DROP COLUMN IF EXISTS oss_coverage,
  DROP COLUMN IF EXISTS oss_faithfulness,
  DROP COLUMN IF EXISTS oss_hallucination,
  DROP COLUMN IF EXISTS oss_ctx_grounding,
  DROP COLUMN IF EXISTS oss_overall;

ALTER TABLE "modelComp"
  ADD COLUMN oss_struct_correct INT DEFAULT 0,
  ADD COLUMN oss_task_success INT DEFAULT 0,
  ADD COLUMN oss_instr_follow INT DEFAULT 0,
  ADD COLUMN oss_coverage INT DEFAULT 0,
  ADD COLUMN oss_faithfulness INT DEFAULT 0,
  ADD COLUMN oss_hallucination INT DEFAULT 0,
  ADD COLUMN oss_ctx_grounding INT DEFAULT 0,
  ADD COLUMN oss_overall INT DEFAULT 0;

-- ============================================================================
-- PART 3: Verify Column Types
-- ============================================================================

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'modelComp'
AND (column_name LIKE 'alpaca_%' OR column_name LIKE 'oss_%')
ORDER BY column_name;

-- ============================================================================
-- PART 4: Create Decimal View (for display purposes)
-- ============================================================================
-- This view automatically converts INT values to decimals when queried

CREATE OR REPLACE VIEW modelComp_with_decimals AS
SELECT 
    id,
    label,
    input,
    actual_output,
    context,
    tuned_alpaca,
    tuned_oss20b,
    -- Alpaca metrics (INT to decimal: value / 10000)
    alpaca_struct_correct / 10000.0 AS alpaca_struct_correct_dec,
    alpaca_task_success / 10000.0 AS alpaca_task_success_dec,
    alpaca_instr_follow / 10000.0 AS alpaca_instr_follow_dec,
    alpaca_coverage / 10000.0 AS alpaca_coverage_dec,
    alpaca_faithfulness / 10000.0 AS alpaca_faithfulness_dec,
    alpaca_hallucination / 10000.0 AS alpaca_hallucination_dec,
    alpaca_ctx_grounding / 10000.0 AS alpaca_ctx_grounding_dec,
    alpaca_overall / 10000.0 AS alpaca_overall_dec,
    -- OSS metrics (INT to decimal: value / 10000)
    oss_struct_correct / 10000.0 AS oss_struct_correct_dec,
    oss_task_success / 10000.0 AS oss_task_success_dec,
    oss_instr_follow / 10000.0 AS oss_instr_follow_dec,
    oss_coverage / 10000.0 AS oss_coverage_dec,
    oss_faithfulness / 10000.0 AS oss_faithfulness_dec,
    oss_hallucination / 10000.0 AS oss_hallucination_dec,
    oss_ctx_grounding / 10000.0 AS oss_ctx_grounding_dec,
    oss_overall / 10000.0 AS oss_overall_dec,
    -- Other columns
    eval_status,
    eval_winner,
    created_at,
    updated_at
FROM "modelComp";

-- You can now query the view like:
-- SELECT * FROM modelComp_with_decimals WHERE label = 'math_logic';

