# Alchemy — Current Tasks

Read this file when you start. Do tasks in order, top to bottom. Skip tasks marked [DONE]. Commit and push after each task.

**Repo:** `C:\Users\monic\Documents\Alchemy_explore` (branch: main)

---

## Task 1: Register NEOSY models in gpu_fleet.yaml
- Edit `config/gpu_fleet.yaml`
- BGE-M3 is already there as warm
- Add siglip2: display_name "SigLIP 2 Image Embedding", backend custom, vram_mb 1500, ram_mb 1500, disk_mb 1500, preferred_gpu 0, default_tier warm, capabilities [embedding, vision]
- Add clap: display_name "CLAP Audio Embedding", backend custom, vram_mb 800, ram_mb 800, disk_mb 800, preferred_gpu 0, default_tier warm, capabilities [embedding, audio]

## Task 2: Add NEOSY settings to Alchemy config
- Edit `config/settings.py`
- Add NeosySettings nested BaseModel class with fields:
  - enabled: bool = True
  - pg_host: str = "localhost"
  - pg_port: int = 5432
  - pg_user: str = "baratza"
  - pg_password: str = "baratza_dev"
  - pg_database: str = "baratza_knowledge"
  - qdrant_host: str = "localhost"
  - qdrant_port: int = 6333
- Add `neosy: NeosySettings = NeosySettings()` to the main Settings class

## Task 3: Mount NEOSY as sub-app in server.py
- Edit `alchemy/server.py`
- In lifespan, after the AlchemyMemory block (~line 327):
  - Create asyncpg pool (host=localhost, port=5432, user=baratza, password=baratza_dev, db=baratza_knowledge)
  - Create QdrantClient(localhost:6333)
  - Store as app.state.neosy_pool and app.state.neosy_qdrant
- Mount NEOSY routes at /v1/neosy/* prefix:
  - Add `sys.path.insert(0, r'C:\Users\monic\BaratzaMemory\src')`
  - Import from baratza.api.routes
- Add cleanup in shutdown
- Add neosy_enabled to /health endpoint

## Task 4: APU stabilization and self-healing
- Audit `alchemy/apu/orchestrator.py` (~761 lines)
- Identify: crash points, VRAM tracking drift, race conditions, recovery gaps
- Fix: add asyncio.Lock for model state transitions
- Fix: periodic VRAM reconciliation vs nvidia-smi
- Fix: rollback on failed model loads
- Fix: startup reconciliation with Ollama actual state
- Add health_check() method
- Create `alchemy/apu/CHANGELOG.md` documenting what was broken and fixed
