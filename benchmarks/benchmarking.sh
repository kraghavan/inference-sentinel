sleep 10
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/v1/inference \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}' | jq '.sentinel.endpoint'
done

# 6. Hard refresh Grafana (Cmd+Shift+R)

# Run all benchmarks
python -m benchmarks.harness --generate --count 50
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