#!/usr/bin/env bash
set -e
source /home/feneti/ai-engineering-camp/myenv/bin/activate
cd "$(dirname "$0")/backend"
exec uvicorn app:app --host 0.0.0.0 --port 8000
