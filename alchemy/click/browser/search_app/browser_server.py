"""
AlchemyBrowser — Search + AI Summary Server
Port: 8055
Search: DuckDuckGo (no API key needed)
AI:     Ollama Qwen3 14B (think:false for speed)
Scrape: trafilatura (clean text extraction)
"""

import asyncio
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import trafilatura
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from pathlib import Path

app = FastAPI(title="AlchemyBrowser Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "qwen3:14b"
SCRAPE_CHARS  = 1500  # max content chars sent to model
SUMMARY_WORDS = 20    # hard max words for AI summary


# ── Serve the HTML ────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

@app.get("/")
async def serve_app():
    return FileResponse(HERE / "AlchemyBrowser.html")


# ── Autocomplete suggestions ──────────────────────────────────────────────────

@app.get("/api/suggest")
async def suggest(q: str):
    """DuckDuckGo autocomplete suggestions."""
    if not q.strip():
        return []
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                "https://duckduckgo.com/ac/",
                params={"q": q, "type": "list"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            data = resp.json()
            # DuckDuckGo returns [query, [suggestions]]
            if isinstance(data, list) and len(data) >= 2:
                return [s["phrase"] for s in data[1] if isinstance(s, dict)]
            return []
    except Exception:
        return []


# ── Search ────────────────────────────────────────────────────────────────────

@app.get("/api/search")
async def search(q: str, count: int = 10):
    if not q.strip():
        raise HTTPException(400, "Empty query")

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q.strip(), max_results=count):
                href   = r.get("href", "")
                domain = _domain(href)
                results.append({
                    "url":         href,
                    "title":       r.get("title", ""),
                    "description": r.get("body", ""),
                    "domain":      domain,
                    "favicon_url": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                })
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")

    return {"query": q, "count": len(results), "results": results}


# ── Summarize ─────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    url:           str
    original_desc: str
    query:         str

@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    content = await _scrape(req.url)

    # Fall back to original description if scrape failed
    source_text = content if content else req.original_desc
    source_text = source_text[:SCRAPE_CHARS]

    prompt = (
        f'Query: "{req.query}"\n\n'
        f"Page excerpt:\n{source_text}\n\n"
        f"Write ONE sentence, maximum {SUMMARY_WORDS} words, that tells the user "
        f"exactly what they will find on this page relevant to their query. "
        f"Be specific. No intro phrases. Output only the sentence, nothing else."
    )

    ai_text = await _ollama(prompt)
    return {"ai_text": ai_text, "scraped": bool(content)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else url

async def _scrape(url: str) -> str:
    """Fetch and extract clean text from a URL. Returns '' on any failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            ) or ""
            return text.strip()
    except Exception:
        return ""

async def _ollama(prompt: str) -> str:
    """Call Ollama Qwen3 14B and return the response text."""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   False,   # top-level — disables CoT for Qwen3
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 60,
                    },
                },
            )
            data = resp.json()
            text = data.get("response", "").strip()
            # Strip any <think>...</think> block Qwen3 may prepend
            import re as _re
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            return text
    except Exception as e:
        return ""


# ── Entry point ───────────────────────────────────────────────────────────────

PORT = 8055

def _kill_stale_server():
    """Kill any process already holding our port so we never get stuck."""
    import subprocess, sys
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if f":{PORT}" in line and "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print(f"  Killed stale process on port {PORT} (PID {pid})")
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    _kill_stale_server()
    print(f"\n  AlchemyBrowser running at  http://localhost:{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
