#!/usr/bin/env python3
"""Add name column to users table"""

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

    print("[INFO] Adding name column to users table...")

    with open('add_name_column.sql', 'r') as f:
        sql = f.read()

    cursor.execute(sql)
    conn.commit()

    print("[SUCCESS] Name column added successfully!")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)
