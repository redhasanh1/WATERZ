#!/usr/bin/env python3
"""Add credits column to users table"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("[ERROR] DATABASE_URL not found")
    exit(1)

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("[INFO] Adding credits column to users table...")

    with open('add_credits_column.sql', 'r') as f:
        sql = f.read()

    cursor.execute(sql)
    conn.commit()

    print("[SUCCESS] Credits column added successfully!")
    print("[INFO] All users now have 5 free credits")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)
