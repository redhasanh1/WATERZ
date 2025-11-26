#!/usr/bin/env python3
import psycopg2
import sys

DATABASE_URL = "postgresql://postgres:xXqaHpjxSUSIXgfPKacAFWkpTURaBzvN@centerbeam.proxy.rlwy.net:49937/railway"
# Note: Server uses postgres.railway.internal:5432 but that's only accessible from Railway's private network
# The public proxy (centerbeam.proxy.rlwy.net:49937) should connect to the SAME database

SQL = """
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS credits INTEGER DEFAULT 5;
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);

-- Email verification columns
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE;

-- Mark existing users as verified (grandfather them in)
UPDATE users SET email_verified = TRUE WHERE email_verified IS NULL;

-- Create verification tokens table
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fast token lookups
CREATE INDEX IF NOT EXISTS idx_email_verification_token ON email_verification_tokens(token);
CREATE INDEX IF NOT EXISTS idx_email_verification_user_id ON email_verification_tokens(user_id);
"""

print("[INFO] Connecting to Railway PostgreSQL...")
try:
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()
    print("[OK] Connected!")

    print("[INFO] Executing migration SQL...")
    cur.execute(SQL)
    conn.commit()
    print("[SUCCESS] Migration completed!")

    print("[INFO] Verifying columns...")
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'users' ORDER BY ordinal_position")
    cols = [row[0] for row in cur.fetchall()]
    print(f"[OK] Columns: {', '.join(cols)}")

    cur.close()
    conn.close()
    print("\n✓ ALL DONE! Registration should now work.")

except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
