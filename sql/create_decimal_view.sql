-- Create a VIEW that displays metrics as decimals
-- This view automatically converts INT8 values (4385) to decimals (0.4385)
-- You can query this view instead of the raw table

CREATE OR REPLACE VIEW inference_results_with_decimals AS
SELECT 
    id,
    input,
    expected_output,
    context,
    actual_output,
    created_at,
    updated_at,
    -- Convert INT8 metrics to decimals by dividing by 10000
    answer_relevancy / 10000.0 AS answer_relevancy,
    contextual_precision / 10000.0 AS contextual_precision,
    contextual_recall / 10000.0 AS contextual_recall,
    contextual_relevancy / 10000.0 AS contextual_relevancy,
    faithfulness / 10000.0 AS faithfulness,
    toxicity / 10000.0 AS toxicity,
    hallucination_rate / 10000.0 AS hallucination_rate,
    overall / 10000.0 AS overall
FROM inference_results;

-- Now you can query like this to see decimal values:
-- SELECT * FROM inference_results_with_decimals;

-- Or use it in the Supabase Table Editor:
-- Just select "inference_results_with_decimals" instead of "inference_results"
