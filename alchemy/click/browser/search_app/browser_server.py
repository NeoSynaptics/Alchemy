"""
AlchemyBrowser — Multi-Engine AI Search Server
Port: 8055
Engines: Google CSE (primary) + Bing (secondary) + DuckDuckGo (fallback)
Fusion:  Reciprocal Rank Fusion (RRF)
AI:      Ollama Qwen3 14B (think:false for speed)
Scrape:  trafilatura (clean text extraction)
"""

import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import trafilatura
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(title="AlchemyBrowser Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "qwen3:14b"
SCRAPE_CHARS  = 1500
SUMMARY_WORDS = 20

# Initialized on startup
_searcher = None


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _init_searcher():
    """Load API keys and initialize multi-engine searcher."""
    global _searcher

    # Try loading keys from Alchemy cloud setup
    try:
        from alchemy.cloud.setup import CloudSetup
        CloudSetup().load_all_keys()
        logger.info("Loaded cloud API keys")
    except Exception:
        logger.info("CloudSetup not available — using env vars directly")

    # Import and init multi-searcher (gracefully handles missing keys)
    try:
        from alchemy.research.multi_search import MultiSearcher
        _searcher = MultiSearcher()
        logger.info("MultiSearcher engines: %s", _searcher.available_engines)
    except ImportError:
        logger.warning("MultiSearcher not available — falling back to DDG only")
        _searcher = None


# ── Serve the HTML ────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

@app.get("/")
async def serve_app():
    return FileResponse(HERE / "AlchemyBrowser.html")


# ── Engine status ─────────────────────────────────────────────────────────────

@app.get("/api/engines")
async def engines():
    """Report which search engines are available."""
    if _searcher:
        return {"engines": _searcher.available_engines}
    return {"engines": ["duckduckgo"]}


# ── Autocomplete suggestions ─────────────────────────────────────────────────

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
            if isinstance(data, list) and len(data) >= 2:
                return [s["phrase"] for s in data[1] if isinstance(s, dict)]
            return []
    except Exception:
        return []


# ── Search (multi-engine + RRF) ──────────────────────────────────────────────

@app.get("/api/search")
async def search(q: str, count: int = 10):
    if not q.strip():
        raise HTTPException(400, "Empty query")

    # Multi-engine path
    if _searcher:
        try:
            ranked = await _searcher.search(q.strip(), max_results=count)
            results = [
                {
                    "url":         r.url,
                    "title":       r.title,
                    "description": r.snippet,
                    "domain":      r.domain,
                    "favicon_url": r.favicon_url,
                    "engines":     r.engines,
                    "rrf_score":   round(r.rrf_score, 5),
                }
                for r in ranked
            ]
            engines_used = _searcher.available_engines
        except Exception as e:
            logger.error("MultiSearcher failed: %s", e)
            raise HTTPException(502, f"Search failed: {e}")
    else:
        # Fallback: DDG only
        results, engines_used = await _search_ddg_fallback(q.strip(), count)

    # Filter non-Latin results (CJK, Arabic, etc.)
    results = [r for r in results if not _has_cjk(r.get("title", "") + r.get("description", ""))]

    return {
        "query": q,
        "count": len(results),
        "engines": engines_used,
        "results": results,
    }


async def _search_ddg_fallback(q: str, count: int) -> tuple[list, list]:
    """Fallback DDG search when MultiSearcher isn't available."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, region="wt-wt", max_results=count):
                href = r.get("href", "")
                domain = _domain(href)
                results.append({
                    "url":         href,
                    "title":       r.get("title", ""),
                    "description": r.get("body", ""),
                    "domain":      domain,
                    "favicon_url": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                    "engines":     ["duckduckgo"],
                    "rrf_score":   0,
                })
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")

    return results, ["duckduckgo"]


# ── Video Search ─────────────────────────────────────────────────────────────

@app.get("/api/videos")
async def video_search(q: str, count: int = 10):
    """Search for videos via DuckDuckGo."""
    if not q.strip():
        raise HTTPException(400, "Empty query")

    try:
        results = await asyncio.to_thread(_video_search_sync, q.strip(), count)
        results = [r for r in results if not _has_cjk(r.get("title", ""))]
        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(502, f"Video search failed: {e}")


def _video_search_sync(q: str, count: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.videos(q, max_results=count))

    results = []
    for v in raw:
        # Parse images dict (comes as string sometimes)
        images = v.get("images", {})
        if isinstance(images, str):
            try:
                import ast
                images = ast.literal_eval(images)
            except Exception:
                images = {}

        # Parse statistics
        stats = v.get("statistics", {})
        if isinstance(stats, str):
            try:
                import ast
                stats = ast.literal_eval(stats)
            except Exception:
                stats = {}

        thumbnail = ""
        if isinstance(images, dict):
            thumbnail = images.get("large", "") or images.get("medium", "") or images.get("small", "")

        results.append({
            "title":       v.get("title", ""),
            "url":         v.get("content", ""),
            "description": v.get("description", ""),
            "duration":    v.get("duration", ""),
            "thumbnail":   thumbnail,
            "embed_url":   v.get("embed_url", ""),
            "publisher":   v.get("publisher", ""),
            "uploader":    v.get("uploader", ""),
            "published":   v.get("published", ""),
            "views":       stats.get("viewCount", 0) if isinstance(stats, dict) else 0,
        })
    return results


# ── News Search ──────────────────────────────────────────────────────────────

@app.get("/api/news")
async def news_search(q: str, count: int = 10):
    """Search for news via DuckDuckGo."""
    if not q.strip():
        raise HTTPException(400, "Empty query")

    try:
        results = await asyncio.to_thread(_news_search_sync, q.strip(), count)
        results = [r for r in results if not _has_cjk(r.get("title", ""))]
        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(502, f"News search failed: {e}")


def _news_search_sync(q: str, count: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.news(q, max_results=count))

    return [
        {
            "title":   n.get("title", ""),
            "url":     n.get("url", ""),
            "body":    n.get("body", ""),
            "source":  n.get("source", ""),
            "date":    n.get("date", ""),
            "image":   n.get("image", ""),
            "domain":  _domain(n.get("url", "")),
        }
        for n in raw
    ]


# ── Image Search ─────────────────────────────────────────────────────────────

@app.get("/api/images")
async def image_search(q: str, count: int = 20):
    """Search for images via DuckDuckGo."""
    if not q.strip():
        raise HTTPException(400, "Empty query")

    try:
        results = await asyncio.to_thread(_image_search_sync, q.strip(), count)
        results = [r for r in results if not _has_cjk(r.get("title", ""))]
        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(502, f"Image search failed: {e}")


def _image_search_sync(q: str, count: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.images(q, max_results=count))
    except Exception:
        return []  # ddgs v9 images sometimes fails

    return [
        {
            "title":     img.get("title", ""),
            "url":       img.get("url", ""),
            "image":     img.get("image", ""),
            "thumbnail": img.get("thumbnail", ""),
            "source":    img.get("source", ""),
            "width":     img.get("width", 0),
            "height":    img.get("height", 0),
        }
        for img in raw
    ]


# ── Summarize ─────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    url:           str
    original_desc: str
    query:         str

@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    content = await _scrape(req.url)

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


# ── Curated Feed (AI picks best pages) ────────────────────────────────────────

class CuratedFeedRequest(BaseModel):
    query:   str
    results: list[dict]   # top search results [{url, title, description}, ...]

@app.post("/api/curated-feed")
async def curated_feed(req: CuratedFeedRequest):
    """Scrape top results, ask AI to pick best pages and generate curated cards."""
    if not req.results:
        return {"cards": []}

    top = req.results[:5]

    # Scrape all pages in parallel
    scrape_tasks = [_scrape(r["url"]) for r in top]
    scraped = await asyncio.gather(*scrape_tasks)

    # Build context for AI
    pages_context = ""
    for i, (r, content) in enumerate(zip(top, scraped)):
        text = (content or r.get("description", ""))[:800]
        pages_context += (
            f"\n--- Page {i+1} ---\n"
            f"URL: {r['url']}\n"
            f"Title: {r['title']}\n"
            f"Content: {text}\n"
        )

    prompt = (
        f'User searched: "{req.query}"\n\n'
        f"Here are the top search results with their page content:\n"
        f"{pages_context}\n\n"
        f"Pick the 3-5 MOST USEFUL pages for this query. For each, output EXACTLY this format "
        f"(one per line, pipe-separated, no extra text):\n"
        f"PAGE_NUMBER|One sentence description (max 15 words)|Two-three sentence preview of what the page covers.\n\n"
        f"Output ONLY the lines, nothing else. No numbering, no intro."
    )

    ai_resp = await _ollama_long(prompt)
    cards = _parse_curated(ai_resp, top)
    return {"cards": cards}


def _parse_curated(ai_text: str, results: list[dict]) -> list[dict]:
    """Parse AI response into curated cards."""
    cards = []
    for line in ai_text.strip().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0].strip()) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                domain = _domain(r["url"])
                cards.append({
                    "url":         r["url"],
                    "title":       r["title"],
                    "domain":      domain,
                    "favicon_url": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                    "description": parts[1].strip(),
                    "preview":     parts[2].strip(),
                })
        except (ValueError, IndexError):
            continue
    return cards


class AIAnswerRequest(BaseModel):
    query:   str
    results: list[dict]   # top search results [{url, title, description}, ...]

@app.post("/api/ai-answer")
async def ai_answer(req: AIAnswerRequest):
    """Generate a direct AI answer from top search results."""
    if not req.results:
        return {"answer": "", "sources": []}

    top = req.results[:5]

    # Scrape pages in parallel
    scrape_tasks = [_scrape(r["url"]) for r in top]
    scraped = await asyncio.gather(*scrape_tasks)

    # Build context
    context = ""
    sources = []
    for i, (r, content) in enumerate(zip(top, scraped)):
        text = (content or r.get("description", ""))[:1000]
        if text:
            context += f"\n[Source {i+1}: {r['title']}]\n{text}\n"
            sources.append({"title": r["title"], "url": r["url"], "domain": _domain(r["url"])})

    if not context.strip():
        return {"answer": "", "sources": []}

    prompt = (
        f'User question: "{req.query}"\n\n'
        f"Sources:\n{context}\n\n"
        f"Write a direct, helpful answer in 3-5 sentences based on the sources above. "
        f"Be specific and factual. No intro phrases like 'Based on...' or 'According to...'. "
        f"Just answer the question directly. If the query is not a question, give a brief "
        f"informative overview. Output only the answer text."
    )

    answer = await _ollama_answer(prompt)
    return {"answer": answer, "sources": sources}


async def _ollama_answer(prompt: str) -> str:
    """Ollama call for AI answer box — medium length."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    },
                },
            )
            data = resp.json()
            text = data.get("response", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
    except Exception:
        return ""


async def _ollama_long(prompt: str) -> str:
    """Ollama call with higher token limit for curated feed."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 300,
                    },
                },
            )
            data = resp.json()
            text = data.get("response", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
    except Exception:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

_CJK_RE = re.compile(r"[\u2E80-\u9FFF\uAC00-\uD7AF\u3040-\u309F\u30A0-\u30FF\u0600-\u06FF]")

def _has_cjk(text: str) -> bool:
    """Check if text contains CJK/Arabic characters (non-Latin results)."""
    return bool(_CJK_RE.search(text))

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
                    "think":   False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 60,
                    },
                },
            )
            data = resp.json()
            text = data.get("response", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
    except Exception:
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
