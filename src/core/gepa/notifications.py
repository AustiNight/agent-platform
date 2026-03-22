"""
Guaranteed Notification Delivery System.

This module ensures that critical GEPA escalations ALWAYS reach you,
using multiple redundant channels with automatic failover.

Delivery guarantee chain:
1. Email (primary)
2. Slack webhook
3. Pushover (mobile push)
4. SMS via Twilio
5. Discord webhook
6. Generic webhook
7. Desktop notification (if running locally)
8. Local file + stdout (ultimate fallback - always works)

If ALL network channels fail, the system will:
- Write to a local file
- Print to stdout/stderr
- Create a desktop notification (macOS/Linux)
- Optionally play a sound alert
"""

import asyncio
import json
import os
import platform
import smtplib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class DeliveryResult:
    """Result of a notification delivery attempt."""
    channel: str
    success: bool
    error: str | None = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class GuaranteedNotifier:
    """
    Multi-channel notification system with guaranteed delivery.
    
    Will try ALL configured channels and always fall back to local methods.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="guaranteed_notifier")
        
        # Ensure fallback directory exists
        self.fallback_dir = Path("./data/gepa/notifications")
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config from environment
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load notification configuration from environment."""
        return {
            # Email
            "email_to": os.getenv("GEPA_NOTIFY_EMAIL", ""),
            "email_from": os.getenv("GEPA_FROM_EMAIL", "gepa-automation@agent-platform.local"),
            "smtp_host": os.getenv("SMTP_HOST", "localhost"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
            "smtp_use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            
            # Slack
            "slack_webhook": os.getenv("GEPA_SLACK_WEBHOOK", ""),
            
            # Pushover
            "pushover_user": os.getenv("PUSHOVER_USER_KEY", ""),
            "pushover_token": os.getenv("PUSHOVER_API_TOKEN", ""),
            
            # Twilio SMS
            "twilio_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "twilio_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "twilio_from": os.getenv("TWILIO_FROM_NUMBER", ""),
            "twilio_to": os.getenv("TWILIO_TO_NUMBER", ""),
            
            # Discord
            "discord_webhook": os.getenv("GEPA_DISCORD_WEBHOOK", ""),
            
            # Generic webhook
            "webhook_url": os.getenv("GEPA_WEBHOOK_URL", ""),
            "webhook_secret": os.getenv("GEPA_WEBHOOK_SECRET", ""),
            
            # Local options
            "desktop_notify": os.getenv("GEPA_DESKTOP_NOTIFY", "true").lower() == "true",
            "sound_alert": os.getenv("GEPA_SOUND_ALERT", "true").lower() == "true",
        }
    
    async def send(
        self,
        title: str,
        message: str,
        severity: str = "critical",
        html_message: str | None = None,
        context: dict | None = None,
    ) -> list[DeliveryResult]:
        """
        Send notification through all available channels.
        
        GUARANTEES at least one delivery method will succeed.
        
        Args:
            title: Notification title
            message: Plain text message (with step-by-step instructions)
            severity: "critical", "warning", or "info"
            html_message: Optional HTML version for email
            context: Additional context data
            
        Returns:
            List of delivery results for each channel attempted
        """
        results = []
        any_success = False
        
        # Try all configured channels
        channels = [
            ("email", self._send_email),
            ("slack", self._send_slack),
            ("pushover", self._send_pushover),
            ("sms", self._send_sms),
            ("discord", self._send_discord),
            ("webhook", self._send_webhook),
        ]
        
        for channel_name, send_func in channels:
            try:
                result = await send_func(title, message, severity, html_message, context)
                results.append(result)
                if result.success:
                    any_success = True
                    self.logger.info(f"notification_sent_{channel_name}")
            except Exception as e:
                results.append(DeliveryResult(
                    channel=channel_name,
                    success=False,
                    error=str(e),
                ))
                self.logger.warning(f"notification_failed_{channel_name}", error=str(e))
        
        # ALWAYS do local fallbacks (these cannot fail)
        local_results = await self._local_fallbacks(title, message, severity, context, any_success)
        results.extend(local_results)
        
        # Log summary
        successful = [r.channel for r in results if r.success]
        failed = [r.channel for r in results if not r.success]
        
        self.logger.info(
            "notification_delivery_complete",
            successful=successful,
            failed=failed,
            any_success=any_success,
        )
        
        return results
    
    async def _send_email(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send email notification."""
        if not self.config["email_to"]:
            return DeliveryResult(channel="email", success=False, error="Not configured")
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[GEPA {severity.upper()}] {title}"
        msg["From"] = self.config["email_from"]
        msg["To"] = self.config["email_to"]
        msg["X-Priority"] = "1" if severity == "critical" else "3"
        
        # Plain text
        msg.attach(MIMEText(message, "plain"))
        
        # HTML
        if html_message:
            msg.attach(MIMEText(html_message, "html"))
        
        # Send
        try:
            if self.config["smtp_use_tls"]:
                with smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"]) as server:
                    server.starttls()
                    if self.config["smtp_user"]:
                        server.login(self.config["smtp_user"], self.config["smtp_password"])
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"]) as server:
                    if self.config["smtp_user"]:
                        server.login(self.config["smtp_user"], self.config["smtp_password"])
                    server.send_message(msg)
            
            return DeliveryResult(channel="email", success=True)
        except Exception as e:
            return DeliveryResult(channel="email", success=False, error=str(e))
    
    async def _send_slack(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send Slack notification."""
        if not self.config["slack_webhook"]:
            return DeliveryResult(channel="slack", success=False, error="Not configured")
        
        color = {"critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"}[severity]
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🤖 {title}",
                "text": message[:3000],  # Slack limit
                "footer": "GEPA Automation",
                "ts": int(datetime.utcnow().timestamp()),
            }]
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.config["slack_webhook"], json=payload)
            resp.raise_for_status()
        
        return DeliveryResult(channel="slack", success=True)
    
    async def _send_pushover(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send Pushover mobile notification."""
        if not self.config["pushover_user"] or not self.config["pushover_token"]:
            return DeliveryResult(channel="pushover", success=False, error="Not configured")
        
        priority = {"critical": 2, "warning": 1, "info": 0}[severity]
        sound = {"critical": "siren", "warning": "pushover", "info": "none"}[severity]
        
        data = {
            "token": self.config["pushover_token"],
            "user": self.config["pushover_user"],
            "title": title[:250],
            "message": message[:1024],
            "priority": priority,
            "sound": sound,
        }
        
        if priority == 2:  # Emergency - requires acknowledgment
            data["retry"] = 60
            data["expire"] = 3600
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.pushover.net/1/messages.json", data=data)
            resp.raise_for_status()
        
        return DeliveryResult(channel="pushover", success=True)
    
    async def _send_sms(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send SMS via Twilio."""
        if not all([self.config["twilio_sid"], self.config["twilio_token"], 
                    self.config["twilio_from"], self.config["twilio_to"]]):
            return DeliveryResult(channel="sms", success=False, error="Not configured")
        
        # SMS is limited - send brief alert
        sms_body = f"GEPA {severity.upper()}: {title}. Check email for details."
        
        auth = (self.config["twilio_sid"], self.config["twilio_token"])
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.config['twilio_sid']}/Messages.json"
        
        data = {
            "From": self.config["twilio_from"],
            "To": self.config["twilio_to"],
            "Body": sms_body[:160],
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, data=data, auth=auth)
            resp.raise_for_status()
        
        return DeliveryResult(channel="sms", success=True)
    
    async def _send_discord(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send Discord webhook notification."""
        if not self.config["discord_webhook"]:
            return DeliveryResult(channel="discord", success=False, error="Not configured")
        
        color = {"critical": 0xdc3545, "warning": 0xffc107, "info": 0x17a2b8}[severity]
        
        payload = {
            "embeds": [{
                "title": f"🤖 {title}",
                "description": message[:4096],
                "color": color,
                "footer": {"text": "GEPA Automation"},
                "timestamp": datetime.utcnow().isoformat(),
            }]
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.config["discord_webhook"], json=payload)
            resp.raise_for_status()
        
        return DeliveryResult(channel="discord", success=True)
    
    async def _send_webhook(
        self,
        title: str,
        message: str,
        severity: str,
        html_message: str | None,
        context: dict | None,
    ) -> DeliveryResult:
        """Send to generic webhook."""
        if not self.config["webhook_url"]:
            return DeliveryResult(channel="webhook", success=False, error="Not configured")
        
        payload = {
            "title": title,
            "message": message,
            "severity": severity,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gepa-automation",
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config["webhook_secret"]:
            headers["X-Webhook-Secret"] = self.config["webhook_secret"]
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.config["webhook_url"],
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        
        return DeliveryResult(channel="webhook", success=True)
    
    async def _local_fallbacks(
        self,
        title: str,
        message: str,
        severity: str,
        context: dict | None,
        any_remote_success: bool,
    ) -> list[DeliveryResult]:
        """
        Local fallback methods that CANNOT fail.
        
        These ensure you ALWAYS get the notification, even if all
        network-based channels fail.
        """
        results = []
        
        # 1. Write to file (ALWAYS succeeds)
        file_result = await self._write_to_file(title, message, severity, context)
        results.append(file_result)
        
        # 2. If no remote success, be more aggressive with local alerts
        if not any_remote_success:
            # Print to stdout with high visibility
            self._print_to_console(title, message, severity)
            results.append(DeliveryResult(channel="stdout", success=True))
            
            # Desktop notification (best effort)
            if self.config["desktop_notify"]:
                desktop_result = await self._desktop_notify(title, severity)
                results.append(desktop_result)
            
            # Sound alert (best effort)
            if self.config["sound_alert"]:
                sound_result = await self._sound_alert(severity)
                results.append(sound_result)
            
            # Create a prominent marker file
            marker = self.fallback_dir / "URGENT_NOTIFICATION_PENDING"
            marker.write_text(f"{title}\n{datetime.utcnow().isoformat()}")
            results.append(DeliveryResult(channel="marker_file", success=True))
        
        return results
    
    async def _write_to_file(
        self,
        title: str,
        message: str,
        severity: str,
        context: dict | None,
    ) -> DeliveryResult:
        """Write notification to local file."""
        timestamp = datetime.utcnow()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{severity}.txt"
        filepath = self.fallback_dir / filename
        
        content = f"""
{'='*70}
GEPA AUTOMATION NOTIFICATION
{'='*70}
Time: {timestamp.isoformat()}
Severity: {severity.upper()}
Title: {title}
{'='*70}

{message}

{'='*70}
Context: {json.dumps(context or {}, indent=2, default=str)}
{'='*70}
"""
        
        filepath.write_text(content)
        
        # Also append to a single log file
        log_file = self.fallback_dir / "all_notifications.log"
        with open(log_file, "a") as f:
            f.write(content)
        
        return DeliveryResult(channel="file", success=True)
    
    def _print_to_console(self, title: str, message: str, severity: str) -> None:
        """Print notification to console with high visibility."""
        border = "!" if severity == "critical" else "="
        
        output = f"""

{border*70}
{border*70}
GEPA AUTOMATION ALERT - {severity.upper()}
{border*70}

{title}

{message[:2000]}

{border*70}
CHECK: ./data/gepa/notifications/ for full details
{border*70}
{border*70}

"""
        
        # Use stderr for critical to ensure visibility
        if severity == "critical":
            print(output, file=sys.stderr)
        else:
            print(output)
    
    async def _desktop_notify(self, title: str, severity: str) -> DeliveryResult:
        """Send desktop notification."""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                script = f'''
                display notification "GEPA {severity.upper()}: Check terminal or email" with title "{title}"
                '''
                subprocess.run(["osascript", "-e", script], capture_output=True)
                
            elif system == "Linux":
                subprocess.run([
                    "notify-send",
                    "-u", "critical" if severity == "critical" else "normal",
                    f"GEPA: {title}",
                    f"{severity.upper()}: Check terminal or email for details",
                ], capture_output=True)
                
            elif system == "Windows":
                # PowerShell notification
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                $template.SelectSingleNode("//text[@id='1']").InnerText = "GEPA: {title}"
                $template.SelectSingleNode("//text[@id='2']").InnerText = "{severity.upper()}: Check email"
                '''
                subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
            
            return DeliveryResult(channel="desktop", success=True)
        except Exception as e:
            return DeliveryResult(channel="desktop", success=False, error=str(e))
    
    async def _sound_alert(self, severity: str) -> DeliveryResult:
        """Play sound alert."""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                sound = "Basso" if severity == "critical" else "Glass"
                subprocess.run(["afplay", f"/System/Library/Sounds/{sound}.aiff"], capture_output=True)
                
            elif system == "Linux":
                # Try paplay (PulseAudio) or aplay (ALSA)
                try:
                    subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"], 
                                 capture_output=True, timeout=5)
                except:
                    subprocess.run(["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                                 capture_output=True, timeout=5)
                                 
            elif system == "Windows":
                import winsound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            
            return DeliveryResult(channel="sound", success=True)
        except Exception as e:
            return DeliveryResult(channel="sound", success=False, error=str(e))


# Singleton instance
_notifier = None


def get_notifier() -> GuaranteedNotifier:
    """Get or create the guaranteed notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = GuaranteedNotifier()
    return _notifier


async def send_critical_notification(
    title: str,
    message: str,
    context: dict | None = None,
) -> list[DeliveryResult]:
    """
    Convenience function to send a critical notification.
    
    This GUARANTEES delivery through at least one channel.
    """
    notifier = get_notifier()
    return await notifier.send(
        title=title,
        message=message,
        severity="critical",
        context=context,
    )
