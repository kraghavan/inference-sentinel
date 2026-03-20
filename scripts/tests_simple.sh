# Terminal 3: Test endpoints

# Health check
curl http://localhost:8000/health | jq

# List models
curl http://localhost:8000/v1/models | jq

# Classification only (no inference)
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My SSN is 123-45-6789"}' | jq

# Safe prompt → should route cloud (falls back to local if no API keys)
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}' | jq

# Sensitive prompt → must route local
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "My credit card is 4111-1111-1111-1111"}]}' | jq

curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, who are you?"}]}' | jq '.sentinel'

# Test
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My credit card is 4111-1111-1111-1111"}' | jq

curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "My credit card is 4111-1111-1111-1111"}]}' 