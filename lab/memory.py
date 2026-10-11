"""Three kinds of memory.
1. Exact run memory   - sha256 of the normalised config; identical experiments are never re-run.
2. Fix memory         - error signature -> structured fix; recurring errors are repaired instantly.
3. Lessons            - free text, retrieved by keyword overlap before a task starts (ChromaDB-free, works offline)."""
import json, math, re
from .db import DB, now, jl

# Keys a fix may touch. A fix is data, validated against this list, never executable code.
PATCHABLE = {"device", "epochs", "lr", "batch_size", "eval_every", "seed"}


class Memory:
    def __init__(self, db: DB):
        self.db = db

    # 1 ---- exact runs
    def find_run(self, requested_hash: str):
        return self.db.one("SELECT * FROM runs WHERE requested_hash=? AND status='success' AND reused_from IS NULL "
                           "ORDER BY id LIMIT 1", (requested_hash,))

    # 2 ---- fixes
    def find_fix(self, sig: str):
        r = self.db.one("SELECT * FROM fix_memory WHERE error_signature=?", (sig,))
        if r:
            r["fix"] = jl(r["fix"], {})
        return r

    def save_fix(self, sig: str, description: str, fix: dict, run_id: int):
        bad = set(fix.get("patch", {})) - PATCHABLE
        if bad:
            raise ValueError(f"fix touches non-patchable keys: {bad}")
        if self.db.one("SELECT 1 FROM fix_memory WHERE error_signature=?", (sig,)):
            self.db.x("UPDATE fix_memory SET success_count=success_count+1 WHERE error_signature=?", (sig,))
        else:
            self.db.x("INSERT INTO fix_memory VALUES(?,?,?,?,?,?)", (sig, description, json.dumps(fix), 1, run_id, now()))

    def bump_fix(self, sig: str):
        self.db.x("UPDATE fix_memory SET success_count=success_count+1 WHERE error_signature=?", (sig,))

    # 3 ---- lessons
    @staticmethod
    def _tok(s):
        return [w for w in re.findall(r"[a-z0-9_.=]+", s.lower()) if len(w) > 2]

    def add_lesson(self, text: str, tags: str = "", run_id: int | None = None):
        self.db.x("INSERT INTO lessons(ts,text,tags,run_id) VALUES(?,?,?,?)", (now(), text, tags, run_id))

    def related_lessons(self, query: str, k=3):
        rows = self.db.all("SELECT * FROM lessons")
        if not rows:  # explicit cold-start path
            return []
        q = set(self._tok(query))
        df = {}
        docs = []
        for r in rows:
            t = set(self._tok(r["text"] + " " + (r["tags"] or "")))
            docs.append((r, t))
            for w in t:
                df[w] = df.get(w, 0) + 1
        n = len(rows)
        scored = []
        for r, t in docs:
            s = sum(math.log(1 + n / df[w]) for w in q & t)
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]
