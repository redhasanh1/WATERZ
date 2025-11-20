-- Migration: Create password_reset_tokens table
-- Date: 2025-01-20
-- Description: Adds password reset functionality with secure tokens

CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_at TIMESTAMP
);

-- Index for fast token lookups
CREATE INDEX IF NOT EXISTS idx_password_reset_token ON password_reset_tokens(token);

-- Index for user_id lookups
CREATE INDEX IF NOT EXISTS idx_password_reset_user_id ON password_reset_tokens(user_id);

-- Index for cleanup of expired tokens
CREATE INDEX IF NOT EXISTS idx_password_reset_expires_at ON password_reset_tokens(expires_at);

-- Comments for documentation
COMMENT ON TABLE password_reset_tokens IS 'Stores password reset tokens for user password recovery';
COMMENT ON COLUMN password_reset_tokens.user_id IS 'Foreign key to users table';
COMMENT ON COLUMN password_reset_tokens.token IS 'Secure random token sent in password reset email';
COMMENT ON COLUMN password_reset_tokens.expires_at IS 'Token expiration timestamp (typically 1 hour from creation)';
COMMENT ON COLUMN password_reset_tokens.used IS 'Flag indicating if token has been used';
COMMENT ON COLUMN password_reset_tokens.used_at IS 'Timestamp when token was used (if applicable)';
