.PHONY: install dev shadow-setup shadow-start shadow-stop shadow-health test check-schemas lock server demo

# Install core dependencies
install:
	pip install -e .

# Install with dev tools
dev:
	pip install -e ".[dev]"

# WSL2 shadow desktop setup
shadow-setup:
	wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/info/GitHub/Alchemy && bash wsl/setup.sh"

# Start shadow desktop
shadow-start:
	wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/info/GitHub/Alchemy && bash wsl/start_shadow.sh"

# Stop shadow desktop
shadow-stop:
	wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/info/GitHub/Alchemy && bash wsl/stop_shadow.sh"

# Check shadow desktop health
shadow-health:
	wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/info/GitHub/Alchemy && bash wsl/health_check.sh"

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

# Run shadow desktop demo
demo:
	python scripts/demo.py
