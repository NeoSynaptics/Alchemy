"""One-time migration: initialize photo_tags + FTS5 + sqlite-vec tables,
and reset the 43 classified photos so VLMWorker re-processes them with
the new JSON prompt (extracts structured tags).

Usage:
    python scripts/migrate_memory_v2.py
"""

import sqlite3
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alchemy.memory.timeline.store import TimelineStore
from alchemy.memory.timeline.vectordb import VectorStore

DB_PATH = Path("D:/AlchemyMemory/timeline.db")


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        return

    # 1. Initialize new tables (photo_tags, FTS5, indexes)
    print("Initializing TimelineStore (creates photo_tags + FTS5)...")
    store = TimelineStore(DB_PATH)
    store.init()

    # 2. Initialize sqlite-vec tables
    print("Initializing VectorStore (creates vec_embeddings + vec_meta)...")
    vectors = VectorStore(str(DB_PATH))
    vectors.init()

    # 3. Rebuild FTS5 index from existing summaries
    print("Rebuilding FTS5 index...")
    store.rebuild_fts()

    # 4. Reset classified photos for re-processing with new JSON prompt
    conn = sqlite3.connect(DB_PATH)
    classified = conn.execute(
        "SELECT id, summary FROM timeline_events "
        "WHERE event_type = 'photo' AND summary != ''"
    ).fetchall()
    print(f"Found {len(classified)} classified photos to re-tag")

    reset_count = 0
    for event_id, summary in classified:
        # Clear summary so VLMWorker picks them up again
        # Save old summary in meta for reference
        row = conn.execute(
            "SELECT meta_json FROM timeline_events WHERE id = ?", (event_id,)
        ).fetchone()
        meta = json.loads(row[0]) if row and row[0] else {}
        meta["old_summary"] = summary
        meta.pop("vlm_status", None)  # Clear status so worker retries

        conn.execute(
            "UPDATE timeline_events SET summary = '', meta_json = ? WHERE id = ?",
            (json.dumps(meta), event_id),
        )
        reset_count += 1

    conn.commit()
    conn.close()

    print(f"Reset {reset_count} photos for re-classification")
    print()
    print("Next steps:")
    print("  1. Start Alchemy server (make server)")
    print("  2. POST /memory/vlm/start to trigger VLM worker")
    print("  3. Photos will be re-classified with JSON prompt -> tags + embeddings")


if __name__ == "__main__":
    main()
