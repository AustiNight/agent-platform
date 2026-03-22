#!/usr/bin/env python3
"""
GEPA Watchdog - Monitors and auto-restarts the automation loop.

This script runs as a separate process and ensures the GEPA
automation loop is always running. If it crashes, the watchdog
restarts it automatically.

Usage:
    python -m src.core.gepa.watchdog
    
Or as a systemd service alongside the main automation.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import structlog

logger = structlog.get_logger()


class GEPAWatchdog:
    """
    Watchdog process for GEPA automation.
    
    Monitors the automation loop and:
    - Restarts if it crashes
    - Sends alert if restarts are too frequent
    - Monitors for stale state (no activity)
    """
    
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.restart_count = 0
        self.last_restart = datetime.utcnow()
        self.max_restarts_per_hour = 5
        self.health_check_interval = 60  # seconds
        self.stale_threshold = timedelta(hours=1)
        
        self._running = True
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("watchdog_shutdown_requested")
        self._running = False
        if self.process:
            self.process.terminate()
    
    def start_automation(self) -> bool:
        """Start the GEPA automation process."""
        try:
            # Kill any existing process
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            
            # Start new process
            cmd = [
                sys.executable,
                "-m", "src.core.gepa.auto_loop",
                "start",
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            
            self.last_restart = datetime.utcnow()
            self.restart_count += 1
            
            logger.info(
                "automation_started",
                pid=self.process.pid,
                restart_count=self.restart_count,
            )
            
            return True
            
        except Exception as e:
            logger.error("automation_start_failed", error=str(e))
            return False
    
    def check_process_health(self) -> bool:
        """Check if the automation process is healthy."""
        if not self.process:
            return False
        
        # Check if process is running
        if self.process.poll() is not None:
            exit_code = self.process.returncode
            logger.warning("automation_exited", exit_code=exit_code)
            return False
        
        # Check for stale state
        state_file = Path("./data/gepa/auto_state.json")
        if state_file.exists():
            mtime = datetime.fromtimestamp(state_file.stat().st_mtime)
            if datetime.utcnow() - mtime > self.stale_threshold:
                logger.warning(
                    "automation_stale",
                    last_update=mtime.isoformat(),
                )
                return False
        
        return True
    
    async def check_restart_limit(self) -> bool:
        """Check if we've exceeded restart limits."""
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        if self.restart_count > self.max_restarts_per_hour and self.last_restart > hour_ago:
            logger.error(
                "restart_limit_exceeded",
                count=self.restart_count,
                limit=self.max_restarts_per_hour,
            )
            
            # Send escalation
            await self._escalate_restart_failure()
            
            # Wait before trying again
            await asyncio.sleep(3600)  # 1 hour
            self.restart_count = 0
            
            return False
        
        # Reset counter if it's been over an hour
        if self.last_restart < hour_ago:
            self.restart_count = 0
        
        return True
    
    async def _escalate_restart_failure(self):
        """Send notification about restart failures."""
        try:
            from src.core.gepa.notifications import send_critical_notification
            
            await send_critical_notification(
                title="GEPA Automation Keeps Crashing",
                message=f"""The GEPA automation process has crashed {self.restart_count} times in the last hour.

The watchdog is pausing restarts for 1 hour to prevent resource exhaustion.

REQUIRED ACTIONS:

1. Check the logs for errors:
   $ journalctl -u gepa-automation -n 100
   $ tail -100 ./data/gepa/logs/gepa-automation.log

2. Run diagnostics:
   $ python -m src.core.gepa.auto_loop diagnose

3. Common causes:
   - API key expired or invalid
   - Database connection issues
   - Out of memory
   - Network connectivity problems

4. After fixing, restart the watchdog:
   $ systemctl restart gepa-watchdog
   # Or: python -m src.core.gepa.watchdog

The system will automatically resume once the issue is resolved.
""",
                context={
                    "restart_count": self.restart_count,
                    "last_restart": self.last_restart.isoformat(),
                },
            )
        except Exception as e:
            logger.error("escalation_failed", error=str(e))
    
    async def run(self):
        """Main watchdog loop."""
        logger.info("watchdog_starting")
        
        # Initial start
        self.start_automation()
        
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self._running:
                    break
                
                # Check health
                if not self.check_process_health():
                    # Check restart limits
                    if await self.check_restart_limit():
                        logger.info("restarting_automation")
                        self.start_automation()
                
            except Exception as e:
                logger.error("watchdog_error", error=str(e))
                await asyncio.sleep(60)
        
        # Cleanup
        if self.process and self.process.poll() is None:
            logger.info("stopping_automation")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        logger.info("watchdog_stopped")


async def main():
    """Entry point."""
    watchdog = GEPAWatchdog()
    await watchdog.run()


if __name__ == "__main__":
    asyncio.run(main())
