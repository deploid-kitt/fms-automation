#!/bin/bash
# FMS Automation Management Script

set -e

PROJECT_DIR="/root/.openclaw/workspace/projects/development/fms-automation"
COMPOSE_FILE="docker-compose.yml"

cd "$PROJECT_DIR"

case "$1" in
    start)
        echo "🚀 Starting FMS Automation..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo ""
        echo "✅ FMS Automation started!"
        echo "   Frontend: https://fms.kitt.deploid.io"
        echo "   API:      https://fms.kitt.deploid.io/api"
        echo "   Docs:     https://fms.kitt.deploid.io/docs"
        echo ""
        echo "   Local ports: backend=8010, frontend=8011"
        ;;
    stop)
        echo "⏹️  Stopping FMS Automation..."
        docker compose -f "$COMPOSE_FILE" down
        ;;
    restart)
        echo "🔄 Restarting FMS Automation..."
        docker compose -f "$COMPOSE_FILE" restart
        ;;
    status)
        echo "📊 FMS Automation Status:"
        docker compose -f "$COMPOSE_FILE" ps
        ;;
    logs)
        shift
        if [ "$1" = "backend" ]; then
            docker compose -f "$COMPOSE_FILE" logs -f backend
        elif [ "$1" = "frontend" ]; then
            docker compose -f "$COMPOSE_FILE" logs -f frontend
        else
            docker compose -f "$COMPOSE_FILE" logs -f
        fi
        ;;
    build)
        echo "🔨 Building FMS Automation..."
        docker compose -f "$COMPOSE_FILE" build
        ;;
    update)
        echo "🔄 Updating FMS Automation..."
        docker compose -f "$COMPOSE_FILE" down
        docker compose -f "$COMPOSE_FILE" build --no-cache
        docker compose -f "$COMPOSE_FILE" up -d
        ;;
    clean)
        echo "🧹 Cleaning up unused Docker resources..."
        docker system prune -f
        ;;
    test)
        echo "🧪 Running backend tests..."
        cd backend
        python -m pytest tests/ -v
        ;;
    shell)
        echo "🐚 Opening shell in backend container..."
        docker compose -f "$COMPOSE_FILE" exec backend /bin/bash
        ;;
    *)
        echo "FMS Automation Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|build|update|clean|test|shell}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - Show and follow logs (optional: backend|frontend)"
        echo "  build    - Build Docker images"
        echo "  update   - Rebuild and restart services"
        echo "  clean    - Clean unused Docker resources"
        echo "  test     - Run backend tests"
        echo "  shell    - Open shell in backend container"
        echo ""
        exit 1
        ;;
esac
