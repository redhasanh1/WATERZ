"""
Emergency B2 cleanup script - Delete ALL files from uploads/ folder
"""
from b2sdk.v2 import B2Api, InMemoryAccountInfo
import os

B2_KEY_ID = os.getenv("B2_KEY_ID", "00539db5c1104b50000000003")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K005384b8lPoBT11wScxkZ2Gx0fszus")
B2_BUCKET = os.getenv("B2_BUCKET", "watermarkz")

print("=" * 60)
print("B2 EMERGENCY CLEANUP - DELETE ALL FILES IN uploads/")
print("=" * 60)

print("\nInitializing B2 API...")
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)

print(f"Getting bucket: {B2_BUCKET}")
bucket = b2_api.get_bucket_by_name(B2_BUCKET)

print("\nListing all files in uploads/ folder...")
files = []
for file_info, _ in bucket.ls(folder_to_list='uploads/', fetch_count=10000):
    files.append({
        'name': file_info.file_name,
        'file_id': file_info.id_,
        'size': file_info.size
    })

print(f"Found {len(files)} files in uploads/ folder")

if len(files) == 0:
    print("No files to delete!")
    exit(0)

print(f"\nDeleting {len(files)} files...")
deleted = 0
errors = 0

for file_info in files:
    try:
        b2_api.delete_file_version(file_info['file_id'], file_info['name'])
        deleted += 1
        print(f"[{deleted}/{len(files)}] Deleted: {file_info['name']} ({file_info['size']} bytes)")
    except Exception as e:
        errors += 1
        print(f"[ERROR] Failed to delete {file_info['name']}: {e}")

print("\n" + "=" * 60)
print(f"CLEANUP COMPLETE")
print(f"Deleted: {deleted} files")
print(f"Errors: {errors} files")
print("=" * 60)
