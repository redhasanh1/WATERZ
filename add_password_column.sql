-- Add password_hash column to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(64);

-- Create index on password_hash for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_password ON users(password_hash);

SELECT 'Password column added successfully!' as message;
