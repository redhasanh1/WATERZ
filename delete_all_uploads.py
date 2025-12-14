"""
Bulk delete ALL files from uploads/ folder in B2
Works around B2 web UI limitations by using API directly
"""
from b2sdk.v2 import B2Api, InMemoryAccountInfo
import os
import time

B2_KEY_ID = os.getenv("B2_KEY_ID", "00539db5c1104b50000000003")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K005384b8lPoBT11wScxkZ2Gx0fszus")
B2_BUCKET = os.getenv("B2_BUCKET", "watermarkz")

print("=" * 60)
print("B2 BULK DELETE - uploads/ folder")
print("=" * 60)

print("\nInitializing B2 API...")
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET)

print("\nListing ALL files in bucket...")
all_files = []
uploads_files = []

# List ALL files (no folder filter - this is the fix!)
for file_info, _ in bucket.ls(fetch_count=10000):
    all_files.append(file_info.file_name)
    if file_info.file_name.startswith('uploads/'):
        uploads_files.append({
            'name': file_info.file_name,
            'file_id': file_info.id_,
            'size': file_info.size
        })

print(f"\n" + "=" * 60)
print(f"Total files in bucket: {len(all_files)}")
print(f"Files in uploads/: {len(uploads_files)}")
print("=" * 60)

if len(uploads_files) == 0:
    print("\n✅ No files to delete in uploads/ folder!")
    print("The folder is already clean.")
    exit(0)

print("\nFiles to be deleted:")
for f in uploads_files[:10]:  # Show first 10
    print(f"  - {f['name']} ({f['size']:,} bytes)")
if len(uploads_files) > 10:
    print(f"  ... and {len(uploads_files) - 10} more files")

# Calculate total size
total_size = sum(f['size'] for f in uploads_files)
print(f"\nTotal size to delete: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

# Ask for confirmation
print("\n" + "=" * 60)
response = input(f"Delete {len(uploads_files)} files from uploads/? (yes/no): ")
if response.lower() != 'yes':
    print("❌ Aborted.")
    exit(0)

# Delete files
print("\n" + "=" * 60)
print("Deleting files...")
print("=" * 60)
deleted = 0
errors = 0

for file_info in uploads_files:
    try:
        b2_api.delete_file_version(file_info['file_id'], file_info['name'])
        deleted += 1
        if deleted % 10 == 0:
            print(f"Progress: {deleted}/{len(uploads_files)} files deleted...")
    except Exception as e:
        errors += 1
        print(f"❌ ERROR deleting {file_info['name']}: {e}")

    # Rate limiting - don't hammer B2 API
    time.sleep(0.1)

print("\n" + "=" * 60)
print("✅ DELETION COMPLETE")
print("=" * 60)
print(f"Successfully deleted: {deleted} files")
print(f"Errors: {errors} files")
print(f"Total cleaned: {total_size/1024/1024:.2f} MB")
print("=" * 60)
