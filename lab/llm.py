"""LLM access through the Elevate gateway (OpenAI Chat Completions format, non-streaming).

- The team key is read from ELEVATE_TEAM_API_KEY only; it is never stored, logged or sent to the browser.
- Every call is metered. The gateway allows 800,000 tokens/day (input+output+reasoning, reset 00:00 UTC);
  LAB_TOKEN_CAP (default 600,000) stops calls before that so a demo can't lock itself out.
- Identical prompts are served from the llm_cache table first, which saves budget and gives LAB_REPLAY=1
  (venue Wi-Fi insurance). With no key, or once the cap is hit, every caller uses its deterministic fallback.
- ask() blocks on the network: call it from a worker thread (asyncio.to_thread), never from the event loop.
"""
import hashlib, json, os, time, urllib.error, urllib.request
from datetime import datetime, timezone
from .db import DB

URL = os.environ.get("ELEVATE_URL", "https://elevate1-0.vercel.app/api/ai")
MODELS = {"gpt-5-mini": "max_completion_tokens", "gpt-5.6-luna": "max_completion_tokens", "deepseek-v4-pro": "max_tokens"}
DEFAULT_MODEL = os.environ.get("LAB_MODEL", "gpt-5-mini")
CAP = int(os.environ.get("LAB_TOKEN_CAP", "600000"))
MAX_REQ_BYTES = 40_000  # gateway limit is 48 KB


def _day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class LLM:
    def __init__(self, db: DB):
        self.db = db
        self.key = os.environ.get("ELEVATE_TEAM_API_KEY", "").strip()
        self.disabled = None  # reason string once the gateway rejects us (401/403)
        self.db.x("CREATE TABLE IF NOT EXISTS llm_usage(day TEXT, model TEXT, tokens INTEGER, calls INTEGER, PRIMARY KEY(day, model))")

    @property
    def available(self):
        if os.environ.get("LAB_REPLAY"):
            return True
        return bool(self.key) and not self.disabled and self.used() < CAP

    def used(self) -> int:
        r = self.db.one("SELECT COALESCE(SUM(tokens),0) t FROM llm_usage WHERE day=?", (_day(),))
        return int(r["t"]) if r else 0

    def usage(self) -> dict:
        return {"configured": bool(self.key), "used": self.used(), "cap": CAP, "gateway_limit": 800_000,
                "day": _day(), "disabled": self.disabled, "model": DEFAULT_MODEL}

    def _meter(self, model, tokens):
        self.db.x("INSERT INTO llm_usage(day,model,tokens,calls) VALUES(?,?,?,1) "
                  "ON CONFLICT(day,model) DO UPDATE SET tokens=tokens+excluded.tokens, calls=calls+1", (_day(), model, tokens))

    def _post(self, body: dict):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)

    def ask(self, system: str, prompt: str, max_tokens=600, model: str | None = None) -> str | None:
        model = model if model in MODELS else DEFAULT_MODEL
        h = hashlib.sha256((model + "\x00" + system + "\x00" + prompt).encode()).hexdigest()[:20]
        row = self.db.one("SELECT response FROM llm_cache WHERE ph=?", (h,))
        if row:
            return row["response"]
        if os.environ.get("LAB_REPLAY") or not self.available:
            return None
        prompt = prompt[:MAX_REQ_BYTES // 2]
        if self.used() + (len(system) + len(prompt)) // 3 + max_tokens * 3 > CAP:
            return None
        # reasoning models spend part of the output limit thinking, so leave room for the answer
        body = {"model": model, "stream": False, MODELS[model]: min(4000, max(1500, max_tokens * 3)),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        for attempt in (1, 2, 3):
            try:
                j = self._post(body)
                self._meter(model, int((j.get("usage") or {}).get("total_tokens") or 0))
                text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if text.strip():
                    self.db.x("INSERT OR REPLACE INTO llm_cache(ph,response) VALUES(?,?)", (h, text))
                    return text
                return None
            except urllib.error.HTTPError as e:
                if (e.code == 429 or e.code >= 500) and attempt < 3:
                    time.sleep(min(10, int(e.headers.get("Retry-After", "3") or 3)))
                    continue
                if e.code in (401, 403):
                    self.disabled = "key rejected (401)" if e.code == 401 else "daily token quota exhausted (403)"
                return None
            except Exception:  # timeout / connection reset: the gateway is occasionally slow, so try again
                if attempt < 3:
                    time.sleep(1)
                    continue
                return None
        return None
