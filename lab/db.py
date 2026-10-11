"""SQLite store: agents, runs, metrics, events, fix memory, lessons, approvals. One file."""
import json, os, sqlite3, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("LAB_DB", ROOT / "data" / "lab.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents(
  id TEXT PRIMARY KEY, name TEXT, version INTEGER DEFAULT 1, parent_id TEXT, parent2_id TEXT,
  capabilities TEXT NOT NULL, system_prompt TEXT NOT NULL, policy TEXT DEFAULT '{}',
  model TEXT, status TEXT DEFAULT 'active', retirement_reason TEXT,
  n_tasks INTEGER DEFAULT 0, avg_reward REAL DEFAULT 0.5, origin TEXT DEFAULT 'seed',
  created_at REAL, retired_at REAL);
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, status TEXT, created_at REAL,
  plan TEXT, report TEXT, findings TEXT);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, request_id INTEGER, label TEXT, requested_hash TEXT,
  requested TEXT, effective TEXT, status TEXT, agent_id TEXT, reused_from INTEGER,
  attempts INTEGER DEFAULT 0, fix_used TEXT, started_at REAL, finished_at REAL,
  wall_s REAL, cpu_s REAL, result TEXT, error_sig TEXT, log_tail TEXT, rating REAL);
CREATE INDEX IF NOT EXISTS runs_hash ON runs(requested_hash);
CREATE TABLE IF NOT EXISTS metrics(
  run_id INTEGER, epoch INTEGER, train_acc REAL, test_acc REAL, train_loss REAL, test_loss REAL);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS fix_memory(
  error_signature TEXT PRIMARY KEY, description TEXT, fix TEXT, success_count INTEGER DEFAULT 1,
  run_id INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS task_rewards(
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, run_id INTEGER, capability TEXT,
  base_reward REAL, rating REAL, reward REAL, note TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT, tags TEXT, run_id INTEGER);
CREATE TABLE IF NOT EXISTS approvals(
  id TEXT PRIMARY KEY, request_id INTEGER, kind TEXT, payload TEXT, status TEXT DEFAULT 'pending',
  created_at REAL, decided_at REAL, decided_via TEXT);
CREATE TABLE IF NOT EXISTS llm_cache(ph TEXT PRIMARY KEY, response TEXT);
"""


class DB:
    def __init__(self, path=DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        self.c.executescript(SCHEMA)

    def x(self, sql, args=()):
        with self.lock:
            cur = self.c.execute(sql, args)
            return cur.lastrowid

    def all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.c.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        with self.lock:
            r = self.c.execute(sql, args).fetchone()
            return dict(r) if r else None


def jl(s, default=None):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def now():
    return time.time()
