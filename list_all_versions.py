"""
List ALL file versions in B2 including hidden/deleted files
"""
from b2sdk.v2 import B2Api, InMemoryAccountInfo
import os

B2_KEY_ID = os.getenv("B2_KEY_ID", "00539db5c1104b50000000003")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K005384b8lPoBT11wScxkZ2Gx0fszus")
B2_BUCKET = os.getenv("B2_BUCKET", "watermarkz")

print("Initializing B2 API...")
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET)

print("\nListing ALL file versions (including hidden/deleted)...")
uploads_count = 0
total_count = 0

# Try with latest_only=False to see ALL versions
for file_version, _ in bucket.ls(latest_only=False, fetch_count=10000):
    total_count += 1
    if file_version.file_name.startswith('uploads/'):
        uploads_count += 1
        print(f"  {file_version.file_name} (action: {file_version.action}, size: {file_version.size})")

print(f"\nTotal file versions: {total_count}")
print(f"Uploads file versions: {uploads_count}")
