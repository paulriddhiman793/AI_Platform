# Makefile for AI Engineering Platform
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev lint format typecheck test test-unit test-integration test-all \
        backend-dev frontend-dev docker-build docker-run docker-stop \
        clean pre-commit ci

# Default target
help:
	@echo "AI Engineering Platform - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with black and ruff"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make pre-commit     - Run pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-frontend  - Run frontend tests"
	@echo ""
	@echo "Development:"
	@echo "  make backend-dev    - Start backend server"
	@echo "  make frontend-dev   - Start frontend dev server"
	@echo "  make dev            - Start both backend and frontend"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-stop    - Stop Docker container"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci             - Run full CI pipeline locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Clean build artifacts"

# Installation
install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"
	cd gui && npm ci

# Code Quality
lint:
	ruff check api/ agents/ tools/ --config pyproject.toml

format:
	ruff format api/ agents/ tools/ --config pyproject.toml
	black api/ agents/ tools/ --config pyproject.toml

typecheck:
	mypy api/ agents/ tools/ --config-file pyproject.toml

pre-commit:
	pre-commit run --all-files

# Testing
test: test-unit test-integration test-frontend

test-unit:
	pytest tests/unit -v --tb=short --cov=api --cov=agents --cov=tools

test-integration:
	pytest tests/integration -v --tb=short

test-frontend:
	cd gui && npm run test:run

test-all: test-unit test-integration test-frontend

# Development
backend-dev:
	export AI_PLATFORM_CLI_INIT=0 && python -m api.main

frontend-dev:
	cd gui && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@$(MAKE) backend-dev & $(MAKE) frontend-dev

# Docker
docker-build:
	docker build -f deploy/hf_space/Dockerfile -t ai-platform .

docker-run:
	docker run -d --name ai-platform \
	  -p 7860:7860 \
	  -e AI_PLATFORM_CLI_INIT=0 \
	  -e GROQ_API_KEY=$${GROQ_API_KEY} \
	  -v $(PWD)/platform_projects:/data/platform_projects \
	  ai-platform

docker-stop:
	docker stop ai-platform && docker rm ai-platform

docker-logs:
	docker logs -f ai-platform

# CI/CD
ci: lint typecheck test-unit test-frontend
	@echo "All CI checks passed!"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov 2>/dev/null || true
	rm -rf gui/dist 2>/dev/null || true

# Security
security:
	bandit -r api/ agents/ tools/ -c pyproject.toml
	semgrep --config=auto api/ agents/ tools/ --error

# Documentation
docs:
	@echo "Documentation commands:"
	@echo "  mkdocs serve    - Serve documentation locally"
	@echo "  mkdocs build    - Build documentation"

# Release
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major