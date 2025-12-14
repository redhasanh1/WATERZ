from b2sdk.v2 import B2Api, InMemoryAccountInfo
import json
import os

B2_KEY_ID = os.getenv("B2_KEY_ID", "00539db5c1104b50000000003")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K005384b8lPoBT11wScxkZ2Gx0fszus")
B2_BUCKET = os.getenv("B2_BUCKET", "watermarkz")

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)

bucket = b2_api.get_bucket_by_name(B2_BUCKET)

print("=" * 60)
print("B2 BUCKET CONFIGURATION")
print("=" * 60)
print(f"Bucket name: {bucket.name}")
print(f"Bucket ID: {bucket.id_}")
print(f"Bucket type: {bucket.type_}")
print(f"\n" + "=" * 60)
print("CURRENT CORS RULES")
print("=" * 60)
print(json.dumps(bucket.cors_rules, indent=2))
