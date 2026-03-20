cd ~/KarthikaRepo/AI/inference-sentinel

# Extract updated benchmarks
cd ~/KarthikaRepo/AI/inference-sentinel

# 1. Extract fixed files
tar -xzf ~/Downloads/dashboard-fixes.tar.gz

# 2. Clear pycache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 3. Clear Prometheus data (removes stale "ollama-native" metrics)
docker-compose down
docker volume rm inference-sentinel_prometheus_data

# 4. Rebuild and restart
docker-compose build --no-cache sentinel
docker-compose up -d

# 5. Wait and generate test requests
sleep 10
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/v1/inference \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}' | jq '.sentinel.endpoint'
done

# 6. Hard refresh Grafana (Cmd+Shift+R)


# Fix NER env var and restart Docker
docker-compose build sentinel
docker-compose up -d --build

# Run all benchmarks
python -m benchmarks.harness --generate --count 200
python -m benchmarks.harness --experiment classification --ner
python -m benchmarks.harness --experiment routing
python -m benchmarks.harness --experiment cost
python -m benchmarks.harness --experiment controller
python -m benchmarks.report

# All experiments (routing, shadow, cost, controller, session)
nohup python -m benchmarks.harness --experiment all > benchmarks_overnight.log 2>&1 &

# Or just session if you want to focus there
nohup python -m benchmarks.harness --experiment session --sessions 10 > session_bench.log 2>&1 &

# Check it's running
jobs -l
tail -f benchmarks_overnight.log