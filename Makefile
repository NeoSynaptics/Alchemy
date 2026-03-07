.PHONY: install dev test check-schemas lock server

# Install core dependencies
install:
	pip install -e .

# Install with dev tools
dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Check schemas.py sync between Alchemy and NEO-TX
check-schemas:
	python scripts/check_schema_sync.py

# Generate pinned dependency lockfile
lock:
	pip-compile pyproject.toml -o requirements-lock.txt --strip-extras

# Run Alchemy server
server:
	uvicorn alchemy.server:app --host 0.0.0.0 --port 8000 --reload
