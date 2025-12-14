"""
List ALL files in B2 bucket to see where they actually are
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

print(f"Getting bucket: {B2_BUCKET}")
bucket = b2_api.get_bucket_by_name(B2_BUCKET)

print("\n" + "=" * 60)
print("ALL FILES IN BUCKET (showing first 100)")
print("=" * 60)

count = 0
for file_info, _ in bucket.ls(fetch_count=100):
    count += 1
    print(f"{count}. {file_info.file_name} ({file_info.size} bytes)")

print("\n" + "=" * 60)
print(f"Total files shown: {count}")
print("=" * 60)
