#!/usr/bin/env python3
"""View latest users from database"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

print("Connecting to database...")

try:
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()
except Exception as e:
    print(f"ERROR: Failed to connect to database: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("LATEST 3 USERS")
print("="*80)

cur.execute('''
    SELECT id, email, name, credits, email_verified, created_at, google_id
    FROM users
    ORDER BY created_at DESC
    LIMIT 3
''')

users = cur.fetchall()

for user in users:
    id, email, name, credits, verified, created, google_id = user
    print(f"\nID:            {id}")
    print(f"Email:         {email}")
    print(f"Name:          {name}")
    print(f"Credits:       {credits}")
    print(f"Verified:      {verified}")
    print(f"Created:       {created}")
    print(f"Google ID:     {google_id or 'N/A'}")
    print("-"*40)

cur.close()
conn.close()

print(f"\nTotal: {len(users)} users shown")
