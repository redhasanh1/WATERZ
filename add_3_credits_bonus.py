#!/usr/bin/env python3
"""
Add 3 FREE Credits Bonus to All Users

This script adds 3 bonus credits to every user in the database as a thank you
for being early supporters during technical difficulties.
"""

import psycopg2

# Database connection details
DB_HOST = "centerbeam.proxy.rlwy.net"
DB_PORT = "49937"
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASSWORD = "xXqaHpjxSUSIXgfPKacAFWkpTURaBzvN"

try:
    # Connect to database
    print(f"[DATABASE] Connecting to PostgreSQL database...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = conn.cursor()

    # Check current total credits before
    cursor.execute("SELECT SUM(credits) FROM users")
    total_before = cursor.fetchone()[0] or 0
    print(f"[DATABASE] Total credits before: {total_before}")

    # Add 3 credits to all users
    print(f"[DATABASE] Adding 3 bonus credits to all users...")
    cursor.execute("UPDATE users SET credits = credits + 3")
    affected_rows = cursor.rowcount

    # Commit the transaction
    conn.commit()

    # Check current total credits after
    cursor.execute("SELECT SUM(credits) FROM users")
    total_after = cursor.fetchone()[0] or 0
    print(f"[DATABASE] Total credits after: {total_after}")

    print(f"\nSUCCESS! Added 3 credits to {affected_rows} users")
    print(f"Total credits distributed: {total_after - total_before}")

    # Close connection
    cursor.close()
    conn.close()

except Exception as e:
    print(f"\nERROR: {e}")
    print(f"\nMake sure:")
    print(f"  1. psycopg2 is installed: pip install psycopg2-binary")
    print(f"  2. Database connection details are correct")
    print(f"  3. You have network access to the database")
