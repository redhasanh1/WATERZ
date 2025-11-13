-- Fix users table schema by dropping and recreating
-- This fixes the "value too long for type character(1)" error

DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);

SELECT 'Users table fixed successfully!' as message;
