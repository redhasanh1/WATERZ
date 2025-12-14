#!/usr/bin/env python3
"""
Setup B2 Lifecycle Rules for Automatic File Deletion

This script configures your B2 bucket to automatically delete files after 1 day.
Zero Class C transaction costs - B2 handles cleanup for free!

Usage:
    python setup_b2_lifecycle.py
"""

import os
from b2sdk.v2 import InMemoryAccountInfo, B2Api

def setup_lifecycle_rules():
    """Configure B2 lifecycle rules for automatic file deletion"""

    # Get B2 credentials from environment
    b2_key_id = os.getenv('B2_KEY_ID')
    b2_app_key = os.getenv('B2_APP_KEY')
    b2_bucket_name = os.getenv('B2_BUCKET', 'watermarkz')

    if not b2_key_id or not b2_app_key:
        print("ERROR: B2_KEY_ID and B2_APP_KEY environment variables must be set")
        print("\nSet them with:")
        print("  export B2_KEY_ID='your_key_id'")
        print("  export B2_APP_KEY='your_app_key'")
        return False

    print(f"[B2] Connecting to B2...")
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", b2_key_id, b2_app_key)

    print(f"[B2] Finding bucket: {b2_bucket_name}")
    bucket = b2_api.get_bucket_by_name(b2_bucket_name)

    # Define lifecycle rules
    lifecycle_rules = [
        {
            'daysFromUploadingToHiding': 1,
            'daysFromHidingToDeleting': 0,
            'fileNamePrefix': 'uploads/'
        },
        {
            'daysFromUploadingToHiding': 1,
            'daysFromHidingToDeleting': 0,
            'fileNamePrefix': 'results/'
        },
        {
            'daysFromUploadingToHiding': 1,
            'daysFromHidingToDeleting': 0,
            'fileNamePrefix': 'masks/'
        }
    ]

    print(f"\n[B2] Configuring lifecycle rules:")
    print(f"  - uploads/  → Delete after 1 day")
    print(f"  - results/  → Delete after 1 day")
    print(f"  - masks/    → Delete after 1 day")

    # Update bucket with lifecycle rules
    bucket.update(
        lifecycle_rules=lifecycle_rules
    )

    print(f"\n✅ SUCCESS! B2 lifecycle rules configured.")
    print(f"\nBenefits:")
    print(f"  - Zero Class C transaction costs (no more list operations!)")
    print(f"  - Automatic cleanup (B2 handles it, not your server)")
    print(f"  - ~$0.15/month storage cost (vs transaction costs)")
    print(f"\nFiles will be automatically deleted after 1 day.")

    return True

if __name__ == '__main__':
    try:
        setup_lifecycle_rules()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\nMake sure:")
        print(f"  1. B2_KEY_ID and B2_APP_KEY are set correctly")
        print(f"  2. You have permission to modify bucket settings")
        print(f"  3. b2sdk is installed: pip install b2sdk")
