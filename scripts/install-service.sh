#!/bin/bash
# Install FMS Automation systemd service

set -e

SERVICE_FILE="fms-automation.service"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Copy service file
cp "$SCRIPT_DIR/$SERVICE_FILE" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service
systemctl enable fms-automation.service

echo "✅ FMS Automation service installed!"
echo ""
echo "Commands:"
echo "   systemctl start fms-automation    - Start the service"
echo "   systemctl stop fms-automation     - Stop the service"
echo "   systemctl status fms-automation   - Check status"
echo "   journalctl -u fms-automation      - View logs"
echo ""
echo "The service will auto-start on boot."
