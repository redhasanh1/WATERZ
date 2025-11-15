-- Add SAM2 jobs table for local processing queue
CREATE TABLE IF NOT EXISTS sam2_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    video_url TEXT NOT NULL,
    video_filename VARCHAR(500),
    points_data JSONB NOT NULL,
    frame_index INTEGER NOT NULL DEFAULT 0,
    video_width INTEGER NOT NULL,
    video_height INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result_url TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Index for fast polling by local worker
CREATE INDEX IF NOT EXISTS idx_sam2_jobs_status ON sam2_jobs(status, created_at);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_sam2_jobs_user_email ON sam2_jobs(user_email, created_at DESC);

-- Index for job_id lookups
CREATE INDEX IF NOT EXISTS idx_sam2_jobs_job_id ON sam2_jobs(job_id);

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_sam2_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
DROP TRIGGER IF EXISTS trigger_sam2_jobs_updated_at ON sam2_jobs;
CREATE TRIGGER trigger_sam2_jobs_updated_at
    BEFORE UPDATE ON sam2_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_sam2_jobs_updated_at();
