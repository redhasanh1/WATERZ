"""Run the email verification database migration."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("[ERROR] DATABASE_URL not found in environment variables!")
    print("Please set DATABASE_URL in your .env file")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("[ERROR] psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)

print(f"[INFO] Connecting to database...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("[INFO] Running email verification migration...")

    # Add email verification columns
    cur.execute('''
        ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE;
    ''')
    print("[OK] Added email_verified column")

    cur.execute('''
        ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_token VARCHAR(64);
    ''')
    print("[OK] Added verification_token column")

    cur.execute('''
        ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_token_expires TIMESTAMP;
    ''')
    print("[OK] Added verification_token_expires column")

    # Create index for faster token lookups
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_users_verification_token ON users(verification_token);
    ''')
    print("[OK] Created verification_token index")

    # Mark existing users as verified (they signed up before verification was required)
    cur.execute('''
        UPDATE users SET email_verified = TRUE WHERE email_verified IS NULL;
    ''')
    print("[OK] Marked existing users as verified")

    cur.close()
    conn.close()

    print("\n[SUCCESS] Email verification migration completed!")
    print("You can now deploy the updated server_production.py")

except Exception as e:
    print(f"[ERROR] Migration failed: {e}")
    sys.exit(1)
