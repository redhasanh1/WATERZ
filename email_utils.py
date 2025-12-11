import os
import requests
from datetime import datetime


# Package display names
PACKAGE_NAMES = {
    'credits_5': 'Starter Pack (5 Credits)',
    'credits_15': 'Basic Pack (15 Credits)',
    'credits_60': 'Pro Pack (60 Credits)',
    'starter': 'Starter Subscription (10 Credits/mo)',
    'pro': 'Pro Subscription (35 Credits/mo)',
}


def send_receipt_email(to_email: str, package_key: str, credits: int, amount_cents: int,
                       currency: str, card_last4: str, card_brand: str, new_balance: int):
    """Send purchase receipt email via Brevo HTTP API."""

    api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "noreply@markremoverai.com")

    if not api_key:
        raise ValueError("Missing BREVO_API_KEY environment variable")

    # Format amount
    amount = amount_cents / 100 if amount_cents else 0
    currency_symbol = '$' if currency.lower() == 'usd' else currency.upper() + ' '
    formatted_amount = f"{currency_symbol}{amount:.2f}"

    # Get package display name
    package_name = PACKAGE_NAMES.get(package_key, package_key)

    # Current date
    purchase_date = datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')

    # Payment method display
    payment_method = f"{card_brand} ****{card_last4}" if card_last4 else "Card"

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
        "to": [{"email": to_email}],
        "subject": "Your MarkRemoverAI Receipt",
        "textContent": f"""
MarkRemoverAI - Purchase Receipt
================================

Date: {purchase_date}

ITEM: {package_name}
AMOUNT: {formatted_amount}

--------------------------------
Total Paid: {formatted_amount}
Payment Method: {payment_method}

Credits Added: {credits}
New Balance: {new_balance} credits
--------------------------------

Thank you for your purchase!

Questions? Contact us at support@markremoverai.com

MarkRemoverAI
https://markremoverai.com
""",
        "htmlContent": f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Arial, Helvetica, sans-serif; background-color: #f4f4f4;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f4f4f4;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);">
                    <!-- Header -->
                    <tr>
                        <td align="center" style="padding: 40px 40px 20px;">
                            <h1 style="margin: 0; color: #667eea; font-size: 28px; font-weight: bold;">MarkRemoverAI</h1>
                            <p style="margin: 10px 0 0; color: #888; font-size: 14px;">support@markremoverai.com</p>
                        </td>
                    </tr>

                    <!-- Receipt Title -->
                    <tr>
                        <td align="center" style="padding: 10px 40px 30px;">
                            <h2 style="margin: 0; color: #333; font-size: 22px; border-bottom: 2px solid #667eea; padding-bottom: 10px; display: inline-block;">RECEIPT</h2>
                        </td>
                    </tr>

                    <!-- Date -->
                    <tr>
                        <td style="padding: 0 40px 20px;">
                            <p style="margin: 0; color: #666; font-size: 14px;"><strong>Date:</strong> {purchase_date}</p>
                        </td>
                    </tr>

                    <!-- Items Table -->
                    <tr>
                        <td style="padding: 0 40px;">
                            <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse: collapse;">
                                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    <td style="padding: 12px 15px; color: #fff; font-weight: bold; font-size: 14px; border-radius: 8px 0 0 0;">Item</td>
                                    <td align="right" style="padding: 12px 15px; color: #fff; font-weight: bold; font-size: 14px; border-radius: 0 8px 0 0;">Amount</td>
                                </tr>
                                <tr style="background-color: #f8f9fa;">
                                    <td style="padding: 15px; color: #333; font-size: 14px; border-bottom: 1px solid #eee;">{package_name}</td>
                                    <td align="right" style="padding: 15px; color: #333; font-size: 14px; border-bottom: 1px solid #eee;">{formatted_amount}</td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Totals -->
                    <tr>
                        <td style="padding: 20px 40px;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td style="padding: 8px 0; color: #333; font-size: 16px; font-weight: bold;">Total Paid:</td>
                                    <td align="right" style="padding: 8px 0; color: #667eea; font-size: 18px; font-weight: bold;">{formatted_amount}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0; color: #666; font-size: 14px;">Payment Method:</td>
                                    <td align="right" style="padding: 8px 0; color: #666; font-size: 14px;">{payment_method}</td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Credits Info -->
                    <tr>
                        <td style="padding: 0 40px 30px;">
                            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; padding: 20px; border: 1px solid rgba(102, 126, 234, 0.2);">
                                <table width="100%" cellspacing="0" cellpadding="0">
                                    <tr>
                                        <td style="color: #333; font-size: 14px; padding: 5px 0;">Credits Added:</td>
                                        <td align="right" style="color: #10b981; font-size: 16px; font-weight: bold; padding: 5px 0;">+{credits}</td>
                                    </tr>
                                    <tr>
                                        <td style="color: #333; font-size: 14px; padding: 5px 0;">New Balance:</td>
                                        <td align="right" style="color: #667eea; font-size: 16px; font-weight: bold; padding: 5px 0;">{new_balance} credits</td>
                                    </tr>
                                </table>
                            </div>
                        </td>
                    </tr>

                    <!-- Thank You -->
                    <tr>
                        <td align="center" style="padding: 0 40px 30px;">
                            <p style="margin: 0; color: #333; font-size: 16px;">Thank you for your purchase!</p>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 20px 40px; background-color: #f8f9fa; border-radius: 0 0 12px 12px;">
                            <p style="margin: 0; color: #999; font-size: 12px; text-align: center;">
                                Questions? Contact us at <a href="mailto:support@markremoverai.com" style="color: #667eea;">support@markremoverai.com</a>
                            </p>
                            <p style="margin: 10px 0 0; color: #999; font-size: 12px; text-align: center;">
                                &copy; 2024 MarkRemoverAI. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    }

    print(f"[EMAIL] Sending receipt to {to_email} for {package_name}")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"[EMAIL SUCCESS] Receipt sent to {to_email}")
        return True
    else:
        error_msg = f"Brevo API error {response.status_code}: {response.text}"
        print(f"[EMAIL ERROR] {error_msg}")
        raise Exception(error_msg)


def send_verification_email(to_email: str, verify_url: str):
    """Send email verification link via Brevo HTTP API."""

    api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "noreply@markremoverai.com")

    print(f"[EMAIL DEBUG] Sending verification email via Brevo")
    print(f"[EMAIL DEBUG] From: {smtp_from}")
    print(f"[EMAIL DEBUG] To: {to_email}")

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
        "subject": "Verify your MarkRemoverAI email",
        "textContent": (
            f"Welcome to MarkRemoverAI!\n\n"
            f"Click the link below to verify your email address:\n\n"
            f"{verify_url}\n\n"
            f"This link expires in 24 hours.\n\n"
            "If you didn't create an account, please ignore this email."
        ),
        "htmlContent": f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Arial, Helvetica, sans-serif; background-color: #f4f4f4;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f4f4f4;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);">
                    <tr>
                        <td align="center" style="padding: 40px 40px 30px;">
                            <h1 style="margin: 0; color: #667eea; font-size: 28px; font-weight: bold;">MarkRemoverAI</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px;">
                            <h2 style="margin: 0 0 20px; color: #333333; font-size: 22px;">Welcome!</h2>
                            <p style="margin: 0 0 25px; color: #555555; font-size: 16px; line-height: 1.6;">
                                Thank you for signing up. Please verify your email address by clicking the button below:
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td align="center" style="padding: 0 40px 30px;">
                            <a href="{verify_url}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; padding: 14px 40px; border-radius: 8px; font-size: 16px; font-weight: bold;">
                                Verify Email Address
                            </a>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px 30px;">
                            <p style="margin: 0 0 15px; color: #777777; font-size: 14px; line-height: 1.5;">
                                Or copy and paste this link into your browser:
                            </p>
                            <p style="margin: 0; color: #667eea; font-size: 13px; word-break: break-all;">
                                {verify_url}
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px 40px;">
                            <p style="margin: 0; color: #999999; font-size: 13px;">
                                This link expires in 24 hours.<br>
                                If you didn't create an account, please ignore this email.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 40px; background-color: #f8f9fa; border-radius: 0 0 12px 12px;">
                            <p style="margin: 0; color: #999999; font-size: 12px; text-align: center;">
                                &copy; 2024 MarkRemoverAI. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    }

    print(f"[EMAIL DEBUG] Sending verification email...")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"[EMAIL SUCCESS] Verification email sent to {to_email}")
        return True
    else:
        error_msg = f"Brevo API error {response.status_code}: {response.text}"
        print(f"[EMAIL ERROR] {error_msg}")
        raise Exception(error_msg)


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
