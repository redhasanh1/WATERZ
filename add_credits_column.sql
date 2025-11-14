-- Add credits column to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS credits INTEGER DEFAULT 5;

-- Update existing users to have 5 free credits if they have NULL
UPDATE users SET credits = 5 WHERE credits IS NULL;

SELECT 'Credits column added successfully!' as message;
