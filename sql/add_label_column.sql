-- Add label column to modelComp table for categorizing inputs
-- Run this in Supabase SQL Editor

ALTER TABLE "modelComp" 
ADD COLUMN IF NOT EXISTS label TEXT;

-- Create index for faster filtering by category
CREATE INDEX IF NOT EXISTS idx_modelcomp_label ON "modelComp" (label);

-- Verify column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'modelComp' AND column_name = 'label';
