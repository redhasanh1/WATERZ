#!/usr/bin/env python3
"""
Setup database for email-only authentication
Runs the SQL migration to create users table
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("[ERROR] DATABASE_URL not found in environment variables!")
    print("Please set DATABASE_URL in your .env file or environment")
    exit(1)

print(f"[INFO] Connecting to database...")
print(f"[INFO] Database URL: {DATABASE_URL[:30]}...")

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("[INFO] Reading SQL migration file...")
    with open('create_users_table.sql', 'r') as f:
        sql = f.read()

    print("[INFO] Executing SQL migration...")
    cursor.execute(sql)
    conn.commit()

    # Verify table was created
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = 'users'
    """)

    result = cursor.fetchone()

    if result:
        print("[SUCCESS] Users table created successfully!")
        print("[INFO] Table: users")
        print("[INFO] Columns: id, email, created_at")
    else:
        print("[WARNING] Table creation query executed but table not found")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] Database setup failed: {e}")
    exit(1)
