#!/bin/bash
# Add FMS subdomain to Caddy configuration with WebSocket support

set -e

CADDY_CONFIG="/etc/caddy/Caddyfile"
BACKUP_FILE="/etc/caddy/Caddyfile.backup.$(date +%Y%m%d_%H%M%S)"

# Backup existing config
cp "$CADDY_CONFIG" "$BACKUP_FILE"
echo "✅ Backed up Caddyfile to $BACKUP_FILE"

# Check if fms.kitt.deploid.io already exists
if grep -q "fms.kitt.deploid.io" "$CADDY_CONFIG"; then
    echo "⚠️  fms.kitt.deploid.io already configured in Caddyfile"
    echo "🔄 Updating configuration for WebSocket support..."
    
    # Remove old config block and add new one
    sed -i '/# FMS Automation/,/^}$/d' "$CADDY_CONFIG"
fi

# Append FMS configuration with WebSocket support
cat >> "$CADDY_CONFIG" << 'EOF'

# FMS Automation - Functional Movement Screen (with Live Analysis)
fms.kitt.deploid.io {
    # WebSocket routes for live analysis
    @websocket {
        header Connection *Upgrade*
        header Upgrade websocket
    }
    
    handle @websocket {
        reverse_proxy 127.0.0.1:8010 {
            header_up X-Real-IP {remote_host}
            # WebSocket specific settings
            header_up Connection {header.Connection}
            header_up Upgrade {header.Upgrade}
        }
    }

    # API routes - proxy to backend
    handle /api/* {
        reverse_proxy 127.0.0.1:8010 {
            header_up X-Real-IP {remote_host}
            # WebSocket support for /api/ws/* routes
            header_up Connection {header.Connection}
            header_up Upgrade {header.Upgrade}
        }
    }

    # Frontend - proxy to nginx
    handle {
        reverse_proxy 127.0.0.1:8011 {
            header_up X-Real-IP {remote_host}
        }
    }

    # Security headers (relaxed for WebSocket)
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Frame-Options "SAMEORIGIN"
        X-Content-Type-Options "nosniff"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # Allow large video uploads
    request_body {
        max_size 500MB
    }
}
EOF

echo "✅ Added FMS configuration with WebSocket support to Caddyfile"

# Reload Caddy
systemctl reload caddy
echo "✅ Caddy reloaded"

echo ""
echo "🎉 FMS Automation v2.0 is now available at: https://fms.kitt.deploid.io"
echo "   Features:"
echo "   - Video upload analysis (as before)"
echo "   - NEW: Live webcam analysis with real-time feedback"
