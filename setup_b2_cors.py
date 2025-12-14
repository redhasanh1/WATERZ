from b2sdk.v2 import B2Api, InMemoryAccountInfo
import json
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

# Define CORS rules with ALL required headers for B2 Native API
cors_rules = [
    {
        "corsRuleName": "allowDirectUploads",
        "allowedOrigins": [
            "https://markremoverai.com"
        ],
        "allowedOperations": [
            "b2_upload_file",
            "b2_upload_part",
            "b2_download_file_by_name"
        ],
        "allowedHeaders": [
            "authorization",
            "content-type",
            "x-bz-file-name",
            "x-bz-content-sha1",
            "x-bz-info-*"
        ],
        "exposeHeaders": [
            "x-bz-content-sha1"
        ],
        "maxAgeSeconds": 3600
    }
]

print("\n" + "=" * 60)
print("APPLYING CORS RULES")
print("=" * 60)
print(json.dumps(cors_rules, indent=2))
print("\n" + "=" * 60)

print(f"\nUpdating bucket '{B2_BUCKET}' with CORS rules...")
bucket.update(cors_rules=cors_rules)

print("✅ CORS rules updated successfully!")
print("\nWait 2-5 minutes for changes to propagate, then test your upload.")
