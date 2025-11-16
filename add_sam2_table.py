#!/usr/bin/env python3
"""
Add SAM2 jobs table to database
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
    with open('add_sam2_jobs_table.sql', 'r') as f:
        sql = f.read()

    print("[INFO] Executing SQL migration...")
    cursor.execute(sql)
    conn.commit()

    # Verify table was created
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = 'sam2_jobs'
    """)

    result = cursor.fetchone()

    if result:
        print("[SUCCESS] sam2_jobs table created successfully!")

        # Show columns
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'sam2_jobs'
            ORDER BY ordinal_position
        """)

        columns = cursor.fetchall()
        print("\n[INFO] Table columns:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")

    else:
        print("[WARNING] Table creation query executed but table not found")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] Database setup failed: {e}")
    exit(1)
