#!/bin/bash
#
# GEPA Automation Setup Script
#
# This script sets up fully automated GEPA optimization.
# Run this once after deployment to enable continuous improvement.
#
# Usage:
#   chmod +x deploy/setup-gepa-automation.sh
#   ./deploy/setup-gepa-automation.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "GEPA Automation Setup"
echo "========================================"
echo ""

# Check if running as root for systemd setup
USE_SYSTEMD=false
if [ "$EUID" -eq 0 ]; then
    USE_SYSTEMD=true
    echo "[✓] Running as root - will set up systemd service"
else
    echo "[!] Running as non-root - will set up user cron job"
    echo "    Run with sudo for systemd service installation"
fi

echo ""

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "[✗] Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "[✓] Python 3 found"

# Check if project is set up
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "[!] .env file not found. Copying from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "[!] IMPORTANT: Edit .env to configure notifications!"
    echo "    At minimum, set GEPA_NOTIFY_EMAIL"
fi
echo "[✓] Configuration file exists"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p "$PROJECT_DIR/data/gepa/traces"
mkdir -p "$PROJECT_DIR/data/gepa/prompts"
mkdir -p "$PROJECT_DIR/data/gepa/runs"
mkdir -p "$PROJECT_DIR/data/gepa/notifications"
mkdir -p "$PROJECT_DIR/data/gepa/logs"
echo "[✓] Data directories created"

# Install GEPA dependencies
echo ""
echo "Installing GEPA dependencies..."
cd "$PROJECT_DIR"

if [ -f "venv/bin/pip" ]; then
    ./venv/bin/pip install -e ".[gepa]" --quiet
else
    pip install -e ".[gepa]" --quiet
fi
echo "[✓] Dependencies installed"

# Test notification system
echo ""
echo "Testing notification system..."
PYTHON_CMD="${PROJECT_DIR}/venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

$PYTHON_CMD << 'PYEOF'
import asyncio
import os
import sys
sys.path.insert(0, os.getcwd())

from src.core.gepa.notifications import get_notifier

async def test():
    notifier = get_notifier()
    results = await notifier.send(
        title="GEPA Setup Test",
        message="This is a test notification to verify your GEPA automation setup is working.\n\nIf you received this, your notification system is configured correctly!",
        severity="info",
    )
    
    successful = [r.channel for r in results if r.success]
    failed = [r.channel for r in results if not r.success]
    
    print(f"[✓] Notification test complete")
    print(f"    Successful channels: {', '.join(successful) or 'none'}")
    if failed:
        print(f"    Failed channels: {', '.join(failed)}")
    
    return len(successful) > 0

asyncio.run(test())
PYEOF

if [ $? -ne 0 ]; then
    echo "[!] Notification test had issues - check your .env configuration"
fi

# Set up systemd or cron
echo ""
if [ "$USE_SYSTEMD" = true ]; then
    echo "Setting up systemd service..."
    
    # Create service file
    cat > /etc/systemd/system/gepa-automation.service << EOF
[Unit]
Description=GEPA Automated Prompt Optimization Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/python -m src.core.gepa.auto_loop start
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    # Reload and enable
    systemctl daemon-reload
    systemctl enable gepa-automation
    systemctl start gepa-automation
    
    echo "[✓] Systemd service installed and started"
    echo "    Check status: systemctl status gepa-automation"
    echo "    View logs: journalctl -u gepa-automation -f"
    
else
    echo "Setting up cron job..."
    
    # Create wrapper script
    WRAPPER_SCRIPT="$PROJECT_DIR/deploy/run-gepa-check.sh"
    cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
cd "$PROJECT_DIR"
source .env 2>/dev/null || true
$PYTHON_CMD -m src.core.gepa.auto_loop status --check-trigger >> "$PROJECT_DIR/data/gepa/logs/cron.log" 2>&1
EOF
    chmod +x "$WRAPPER_SCRIPT"
    
    # Add to crontab (every 5 minutes)
    CRON_LINE="*/5 * * * * $WRAPPER_SCRIPT"
    
    # Check if already in crontab
    if crontab -l 2>/dev/null | grep -q "gepa-check"; then
        echo "[!] GEPA cron job already exists"
    else
        (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
        echo "[✓] Cron job installed (runs every 5 minutes)"
    fi
    
    echo "    View crontab: crontab -l"
    echo "    View logs: tail -f $PROJECT_DIR/data/gepa/logs/cron.log"
fi

# Create convenience commands
echo ""
echo "Creating convenience commands..."

cat > "$PROJECT_DIR/gepa" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD="./venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

case "$1" in
    status)
        $PYTHON_CMD -m src.core.gepa.auto_loop status
        ;;
    diagnose)
        $PYTHON_CMD -m src.core.gepa.auto_loop diagnose
        ;;
    dashboard)
        echo "Starting dashboard at http://localhost:8082"
        $PYTHON_CMD -m src.core.gepa.dashboard
        ;;
    optimize)
        AGENT_TYPE="${2:-browser}"
        echo "Triggering optimization for $AGENT_TYPE..."
        $PYTHON_CMD -c "
import asyncio
from src.core.gepa.auto_loop import GEPAAutomationLoop
loop = GEPAAutomationLoop()
asyncio.run(loop._run_optimization('$AGENT_TYPE', 'Manual trigger'))
"
        ;;
    traces)
        echo "Recent traces:"
        ls -la ./data/gepa/traces/ | tail -20
        ;;
    notifications)
        echo "Recent notifications:"
        ls -la ./data/gepa/notifications/ | tail -10
        cat ./data/gepa/notifications/*.txt 2>/dev/null | tail -100
        ;;
    test-notify)
        echo "Sending test notification..."
        $PYTHON_CMD -c "
import asyncio
from src.core.gepa.notifications import send_critical_notification
asyncio.run(send_critical_notification(
    title='Test Notification',
    message='This is a test of the GEPA notification system.\\n\\nIf you see this, notifications are working!',
))
"
        ;;
    help|--help|-h|"")
        echo "GEPA CLI - Automated Prompt Optimization"
        echo ""
        echo "Usage: ./gepa <command>"
        echo ""
        echo "Commands:"
        echo "  status        Show current GEPA status"
        echo "  diagnose      Run diagnostics"
        echo "  dashboard     Start web dashboard"
        echo "  optimize      Trigger optimization (optional: agent type)"
        echo "  traces        Show recent execution traces"
        echo "  notifications Show recent notifications"
        echo "  test-notify   Send a test notification"
        echo "  help          Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './gepa help' for usage"
        exit 1
        ;;
esac
EOF
chmod +x "$PROJECT_DIR/gepa"
echo "[✓] Created ./gepa convenience script"

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "GEPA automation is now active. Here's what happens automatically:"
echo ""
echo "  ✅ Every task execution is traced"
echo "  ✅ Traces accumulate until optimization triggers"
echo "  ✅ GEPA evolves prompts based on execution feedback"
echo "  ✅ Improved prompts are applied without restart"
echo "  ✅ You're notified ONLY when human action is needed"
echo ""
echo "Quick commands:"
echo "  ./gepa status       - Check current state"
echo "  ./gepa diagnose     - Run diagnostics"
echo "  ./gepa dashboard    - View web dashboard"
echo "  ./gepa test-notify  - Test notifications"
echo ""
echo "IMPORTANT: Verify your notification settings in .env:"
echo "  GEPA_NOTIFY_EMAIL=your-email@example.com"
echo ""
echo "The system will notify you with detailed step-by-step"
echo "instructions whenever human intervention is truly required."
echo ""
