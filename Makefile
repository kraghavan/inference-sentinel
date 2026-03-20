.PHONY: help install install-dev test test-unit test-integration test-ollama lint format typecheck run run-docker docker-build docker-up docker-down clean

# Default target
help:
	@echo "inference-sentinel development commands:"
	@echo ""
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests (excluding Ollama tests)"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests (requires Ollama)"
	@echo "  test-ollama    Run tests that require Ollama"
	@echo "  lint           Run linter (ruff)"
	@echo "  format         Format code (ruff)"
	@echo "  typecheck      Run type checker (mypy)"
	@echo "  run            Run the application locally"
	@echo ""
	@echo "Docker commands:"
	@echo "  docker-setup     First-time setup (pulls 3GB model, takes ~5 min)"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      Follow sentinel logs"
	@echo "  docker-clean     Stop and remove volumes"
	@echo ""
	@echo "  clean          Remove build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,cloud]"

# Testing
test:
	pytest -v --ignore=tests/integration -m "not ollama"

test-unit:
	pytest -v tests/unit/

test-integration:
	pytest -v tests/integration/

test-ollama:
	pytest -v -m ollama

test-all:
	pytest -v

# Code quality
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

# Running
run:
	python -m sentinel.main

run-reload:
	SENTINEL_DEBUG=true python -m sentinel.main

# Docker
docker-build:
	docker build -t inference-sentinel:latest .

docker-up:
	docker-compose up -d

docker-up-build:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f sentinel

docker-logs-all:
	docker-compose logs -f

docker-ollama-status:
	docker exec inference-sentinel-ollama-1 ollama list

docker-clean:
	docker-compose down -v --remove-orphans

docker-setup: docker-up
	@echo "First-time setup: model download takes 3-5 minutes..."
	@echo "Run 'docker ps' to check status. Wait for ollama to show 'healthy'."
	@echo "Then run 'make docker-logs' to see sentinel logs."

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development helpers
ollama-serve:
	ollama serve

ollama-pull:
	ollama pull gemma3:4b

ollama-status:
	curl -s http://localhost:11434/api/tags | python -m json.tool
