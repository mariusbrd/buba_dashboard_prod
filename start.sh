#!/bin/bash
# Startup script for GVB Dashboard
# Ensures preload runs once before starting gunicorn workers

set -e

echo "🚀 GVB Dashboard Startup"
echo "========================"

# Run preload once (before workers start)
echo "📥 Running startup preloads..."
python3 -c "
import sys
sys.path.insert(0, '/app')
from app import run_startup_preloads
run_startup_preloads()
print('✅ Preload complete')
"

# Start gunicorn with multiple workers
echo "🌐 Starting Gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers 2 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:server
