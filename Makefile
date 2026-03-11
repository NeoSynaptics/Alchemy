.PHONY: install dev test check-schemas lock server ui-install ui-build ui-dev

# Install core dependencies
install:
	pip install -e .

# Install with dev tools (Python)
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

# Install UI dependencies
ui-install:
	cd ui && npm install

# Build React UI for production (output: ui/dist/)
ui-build:
	cd ui && npm run build

# Run React UI dev server (port 5173, proxies API to :8000)
ui-dev:
	cd ui && npm run dev
