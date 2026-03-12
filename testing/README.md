# Testing Hub

Centralized test results for both **Alchemy** and **BaratzaMemory** repos.

## How It Works

1. **Run tests** — `bash testing/run_tests.sh` (or run per-repo manually)
2. **Results land here** — `testing/results/alchemy_latest.md` and `testing/results/baratza_latest.md`
3. **Claude reads results** — distills failures into `TESTING_TODO.md`
4. **Fix, re-run, repeat**

## Manual Runs

```bash
# Alchemy (from repo root)
pytest tests/ --tb=short -q 2>&1 | tee testing/results/alchemy_latest.txt

# BaratzaMemory (from BaratzaMemory root)
cd ~/BaratzaMemory
PYTHONPATH=src pytest tests/ --tb=short -q 2>&1 | tee ~/Documents/Alchemy_explore/testing/results/baratza_latest.txt
```

## For Claude

Read `TESTING_TODO.md` for the current list of failures to fix. After fixing, re-run tests and update results.
