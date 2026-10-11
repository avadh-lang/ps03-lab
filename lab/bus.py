"""Event bus. Every decision/action is an event: persisted (except high-rate metrics) and fanned out to WebSockets."""
import asyncio, json
from .db import DB, now


class Bus:
    def __init__(self, db: DB):
        self.db = db
        self.subs: set[asyncio.Queue] = set()
        self.loop = None  # learned from the first emit made on the event loop

    def emit(self, type_: str, **payload):
        ev = {"type": type_, "ts": now(), **payload}
        if type_ != "metric":  # curves are rebuilt from the metrics table on reconnect
            ev["id"] = self.db.x("INSERT INTO events(ts,type,payload) VALUES(?,?,?)", (ev["ts"], type_, json.dumps(payload)))
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            self.loop = running
            self._fanout(ev)
        elif self.loop is not None:  # emitted from a worker thread (LLM calls run off the loop)
            self.loop.call_soon_threadsafe(self._fanout, ev)
        else:
            self._fanout(ev)
        return ev

    def _fanout(self, ev):
        for q in list(self.subs):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                self.subs.discard(q)  # slow client: it will resync via /api/state

    def subscribe(self):
        q = asyncio.Queue(maxsize=2000)
        self.subs.add(q)
        return q

    def unsubscribe(self, q):
        self.subs.discard(q)

    def since(self, last_id: int, limit=500):
        rows = self.db.all("SELECT id,ts,type,payload FROM events WHERE id>? ORDER BY id LIMIT ?", (last_id, limit))
        return [{"id": r["id"], "ts": r["ts"], "type": r["type"], **json.loads(r["payload"])} for r in rows]
