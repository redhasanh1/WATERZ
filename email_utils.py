import os
import ssl
import smtplib
from email.message import EmailMessage

def send_reset_email(to_email: str, reset_url: str):
    """Send password reset email via SMTP."""
    msg = EmailMessage()
    msg["Subject"] = "Reset your MarkRemoverAI password"
    msg["From"] = os.getenv("SMTP_FROM")
    msg["To"] = to_email
    msg.set_content(
        f"Click the link below to reset your password (expires in 1 hour):\n\n"
        f"{reset_url}\n\n"
        "If you didn't request this, please ignore this email."
    )

    with smtplib.SMTP_SSL(
        os.getenv("SMTP_SERVER"),
        int(os.getenv("SMTP_PORT", "465")),
        context=ssl.create_default_context(),
    ) as server:
        server.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
        server.send_message(msg)

    print(f"[EMAIL] Reset link sent to {to_email}")
