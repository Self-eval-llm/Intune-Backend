-- SQL script to create the Supabase table for storing inference results
-- Run this in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS inference_results (
    id BIGSERIAL PRIMARY KEY,
    input TEXT NOT NULL,
    expected_output TEXT NOT NULL,
    context JSONB DEFAULT '[]'::jsonb,
    actual_output TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on created_at for faster queries
CREATE INDEX IF NOT EXISTS idx_inference_results_created_at ON inference_results(created_at DESC);

-- Enable Row Level Security (optional, but recommended)
ALTER TABLE inference_results ENABLE ROW LEVEL SECURITY;

-- Create a policy that allows all operations (adjust as needed for your security requirements)
CREATE POLICY "Enable all access for authenticated users" ON inference_results
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Or if you want to allow anonymous access (for development only):
-- CREATE POLICY "Enable all access for all users" ON inference_results
--     FOR ALL
--     USING (true)
--     WITH CHECK (true);
