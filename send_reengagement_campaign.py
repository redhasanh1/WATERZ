#!/usr/bin/env python3
"""
Re-Engagement Email Campaign Sender

Sends personalized re-engagement emails to users via Brevo API.
Includes test mode to send to 4 users first before sending to all.

Usage:
    python send_reengagement_campaign.py --test    # Send to 4 test users
    python send_reengagement_campaign.py --all     # Send to all 43 users
"""

import os
import sys
import csv
import smtplib
import time
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Brevo SMTP Configuration
SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
SMTP_USERNAME = "9c3ca4001@smtp-brevo.com"
SMTP_PASSWORD = "xsmtpsib-798b8db725baf9ad346d32e4ccae5bdcfc64f7c03d059b5aef8e6f82fa5da30b-IB5SjcNk2AV6NZ1f"
SMTP_FROM = "support@markremoverai.com"

# Test users list
TEST_USERS = [
    "humblewoslayer@gmail.com",
    "chw2017@gmail.com",
    "leeroybentz@gmail.com",
    "cryptocoins0@yahoo.com"
]

def load_email_template():
    """Load HTML email template"""
    with open('email_template_reengagement.html', 'r', encoding='utf-8') as f:
        return f.read()

def send_email_smtp(to_email, to_name, html_content):
    """Send email via Brevo SMTP with Gmail Primary inbox optimization"""
    try:
        # Handle unicode in name - use fallback if needed
        try:
            safe_name = to_name.encode('ascii').decode('ascii')
        except:
            safe_name = 'there'

        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Sorry - MarkRemoverAI"
        msg['From'] = f"MarkRemoverAI <{SMTP_FROM}>"
        msg['To'] = to_email
        msg['Reply-To'] = SMTP_FROM
        msg['X-Entity-Ref-ID'] = str(uuid.uuid4())

        # Plain text (minimal, personal)
        text_content = f"""Hi {safe_name},

We messed up—our platform had some technical issues that prevented it from working properly. We're truly sorry.

The good news: everything is fixed now and working perfectly. As a thank you for giving us a chance, we've added 3 free credits to your account.

Log in here: https://markremoverai.com/login

Sorry,
Mark"""

        # HTML mirror (simple, no fancy styling)
        minimal_html = f"""<html><body style="font:14px Arial;color:#000">
<p>Hi {safe_name},</p>
<p>We messed up—our platform had some technical issues that prevented it from working properly. We're truly sorry.</p>
<p>The good news: everything is fixed now and working perfectly. As a thank you for giving us a chance, we've added 3 free credits to your account.</p>
<p><a href=https://markremoverai.com/login>markremoverai.com/login</a></p>
<p>Sorry,<br>Mark</p>
</body></html>"""

        # Attach parts - PLAIN TEXT FIRST (critical for Gmail Primary inbox)
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(minimal_html, 'html')
        msg.attach(part1)
        msg.attach(part2)

        # Send via SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"  [SUCCESS] Email sent to {safe_name} ({to_email})")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to send to {to_email}: {str(e)}")
        return False

def send_campaign(test_mode=False):
    """Send re-engagement campaign emails"""
    print(f"\n{'='*60}")
    print(f"  MarkRemoverAI Re-Engagement Email Campaign")
    print(f"{'='*60}\n")

    # Load template
    print("[1/3] Loading email template...")
    template = load_email_template()
    print("      Template loaded successfully!")

    # Load users
    print("\n[2/3] Loading user list...")
    users_to_email = []

    with open('users_email_list.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row['email']
            name = row['name']

            # Filter based on mode
            if test_mode:
                if email in TEST_USERS:
                    users_to_email.append({'email': email, 'name': name})
            else:
                users_to_email.append({'email': email, 'name': name})

    if test_mode:
        print(f"      TEST MODE: Found {len(users_to_email)} test users")
    else:
        print(f"      PRODUCTION MODE: Found {len(users_to_email)} users")

    # Send emails
    print(f"\n[3/3] Sending emails...")
    success_count = 0
    fail_count = 0

    for i, user in enumerate(users_to_email, 1):
        # Only send to user 31 (the one that failed)
        if not test_mode and i != 31:
            continue

        email = user['email']
        name = user['name']

        try:
            print(f"\n  [{i}/{len(users_to_email)}] Sending to {name}...")
        except UnicodeEncodeError:
            print(f"\n  [{i}/{len(users_to_email)}] Sending to {email}...")

        # Personalize template (handle unicode in name)
        try:
            personalized_html = template.replace('{name}', name)
        except:
            personalized_html = template.replace('{name}', 'there')

        # Send email (with Gmail bypass optimizations)
        if send_email_smtp(email, name, personalized_html):
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting - wait 0.5 seconds between emails
        if i < len(users_to_email):
            time.sleep(0.5)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Campaign Summary")
    print(f"{'='*60}")
    print(f"  Total Sent:    {success_count}")
    print(f"  Failed:        {fail_count}")
    print(f"  Success Rate:  {success_count / len(users_to_email) * 100:.1f}%")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python send_reengagement_campaign.py --test    # Send to 4 test users")
        print("  python send_reengagement_campaign.py --all     # Send to all 43 users")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == '--test':
        print("\nTEST MODE: Sending to 4 test users only")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() == 'y':
            send_campaign(test_mode=True)
    elif mode == '--all':
        print("\nPRODUCTION MODE: Sending to ALL 43 users")
        confirm = input("Are you sure? This will send to everyone. (y/n): ")
        if confirm.lower() == 'y':
            send_campaign(test_mode=False)
    else:
        print(f"Unknown mode: {mode}")
        print("Use --test or --all")
