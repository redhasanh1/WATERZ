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
