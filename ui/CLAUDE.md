# AlchemyFrontDev — Safety Boundary

## CRITICAL RULES
- **NEVER modify files outside of `ui/`** — the Python backend in `alchemy/`, `config/`, `tests/`, `scripts/` is OFF LIMITS
- This is a **read-only frontend** that consumes the Alchemy FastAPI on port 8000
- All API calls go through the Vite proxy (`/api` → `localhost:8000`)
- Settings displayed here are informational — config changes happen through the API, not by editing Python files

## Stack
- Vite + React + TypeScript
- Tailwind CSS v4
- shadcn/ui components
- React Router for navigation

## Dev
```bash
cd ui && npm run dev   # starts on :5173, proxies /api to :8000
cd ui && npm run build # production build to ui/dist/
```

## Structure
```
ui/src/
  components/ui/      — shadcn/ui primitives (do not edit manually)
  components/settings/ — settings panel sections
  hooks/              — API hooks (read-only fetch from Alchemy API)
  pages/              — route pages
  lib/                — utilities
```
