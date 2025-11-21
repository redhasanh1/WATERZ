import os
import ssl
import smtplib
from email.message import EmailMessage

def send_reset_email(to_email: str, reset_url: str):
    """Send password reset email via SMTP."""

    # Debug logging
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "465")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM")

    print(f"[EMAIL DEBUG] Server: {smtp_server}")
    print(f"[EMAIL DEBUG] Port: {smtp_port}")
    print(f"[EMAIL DEBUG] Username: {smtp_username}")
    print(f"[EMAIL DEBUG] Password: {'*' * len(smtp_password) if smtp_password else 'NOT SET'}")
    print(f"[EMAIL DEBUG] From: {smtp_from}")
    print(f"[EMAIL DEBUG] To: {to_email}")

    if not all([smtp_server, smtp_port, smtp_username, smtp_password, smtp_from]):
        missing = []
        if not smtp_server: missing.append("SMTP_SERVER")
        if not smtp_port: missing.append("SMTP_PORT")
        if not smtp_username: missing.append("SMTP_USERNAME")
        if not smtp_password: missing.append("SMTP_PASSWORD")
        if not smtp_from: missing.append("SMTP_FROM")
        raise ValueError(f"Missing SMTP environment variables: {', '.join(missing)}")

    msg = EmailMessage()
    msg["Subject"] = "Reset your MarkRemoverAI password"
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(
        f"Click the link below to reset your password (expires in 1 hour):\n\n"
        f"{reset_url}\n\n"
        "If you didn't request this, please ignore this email."
    )

    print(f"[EMAIL DEBUG] Connecting to {smtp_server}:{smtp_port}...")

    port = int(smtp_port)

    # Port 465 uses SSL, Port 587 uses STARTTLS
    if port == 465:
        with smtplib.SMTP_SSL(
            smtp_server,
            port,
            context=ssl.create_default_context(),
        ) as server:
            print(f"[EMAIL DEBUG] Logging in as {smtp_username}...")
            server.login(smtp_username, smtp_password)
            print(f"[EMAIL DEBUG] Sending message...")
            server.send_message(msg)
    else:
        # Port 587 with STARTTLS
        with smtplib.SMTP(smtp_server, port) as server:
            print(f"[EMAIL DEBUG] Starting TLS...")
            server.starttls(context=ssl.create_default_context())
            print(f"[EMAIL DEBUG] Logging in as {smtp_username}...")
            server.login(smtp_username, smtp_password)
            print(f"[EMAIL DEBUG] Sending message...")
            server.send_message(msg)

    print(f"[EMAIL SUCCESS] Reset link sent to {to_email}")
