import os
import requests

def send_reset_email(to_email: str, reset_url: str):
    """Send password reset email via Brevo HTTP API."""

    api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "noreply@markremoverai.com")

    print(f"[EMAIL DEBUG] Using Brevo HTTP API")
    print(f"[EMAIL DEBUG] From: {smtp_from}")
    print(f"[EMAIL DEBUG] To: {to_email}")
    print(f"[EMAIL DEBUG] API Key: {'*' * 20}...{api_key[-8:] if api_key else 'NOT SET'}")

    if not api_key:
        raise ValueError("Missing BREVO_API_KEY environment variable")

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    payload = {
        "sender": {
            "name": "MarkRemoverAI",
            "email": smtp_from
        },
        "to": [
            {
                "email": to_email
            }
        ],
        "subject": "Reset your MarkRemoverAI password",
        "textContent": (
            f"Click the link below to reset your password (expires in 1 hour):\n\n"
            f"{reset_url}\n\n"
            "If you didn't request this, please ignore this email."
        )
    }

    print(f"[EMAIL DEBUG] Sending via Brevo API...")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"[EMAIL SUCCESS] Reset link sent to {to_email}")
        return True
    else:
        error_msg = f"Brevo API error {response.status_code}: {response.text}"
        print(f"[EMAIL ERROR] {error_msg}")
        raise Exception(error_msg)
