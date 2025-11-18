-- Migration: Add authentication columns to users table
-- This fixes the "column password_hash does not exist" error

-- Add password_hash column for email/password authentication
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);

-- Add name column for user display name
ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255);

-- Add credits column for credit system
ALTER TABLE users ADD COLUMN IF NOT EXISTS credits INTEGER DEFAULT 5;

-- Add google_id column for Google OAuth
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id VARCHAR(255);

-- Add subscription columns for Stripe integration
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255);

-- Create index on google_id for faster OAuth lookups
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);

-- Show table structure after migration
\d users;
