#!/bin/bash
# Install FMS CLI command

set -e

CLI_PATH="/usr/local/bin/fms"

cat > "$CLI_PATH" << 'EOF'
#!/bin/bash
# FMS Automation CLI
exec /root/.openclaw/workspace/projects/development/fms-automation/manage.sh "$@"
EOF

chmod +x "$CLI_PATH"

echo "✅ FMS CLI installed!"
echo "   Run 'fms' from anywhere to manage the application"
echo ""
echo "Examples:"
echo "   fms start    - Start all services"
echo "   fms status   - Check status"
echo "   fms logs     - View logs"
