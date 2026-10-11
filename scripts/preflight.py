"""One-command readiness check. Prints PASS/WARN/FAIL per item; exits 1 on any FAIL. Spends no LLM tokens and never prints the key."""
import importlib, os, shutil, socket, sqlite3, sys, tempfile, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bad = 0


def say(level, what, detail=""):
    global bad
    bad += level == "FAIL"
    print(f"  {level:<4}  {what}" + (f"  ({detail})" if detail else ""))


print("LabDex preflight")
say("PASS" if sys.version_info >= (3, 10) else "FAIL", "python >= 3.10", sys.version.split()[0])
for mod in ("fastapi", "uvicorn", "pydantic", "torch", "qrcode", "websockets"):
    try:
        m = importlib.import_module(mod)
        say("PASS", f"import {mod}", getattr(m, "__version__", ""))
    except Exception as e:
        say("FAIL", f"import {mod}", type(e).__name__)

try:
    import torch
    x = torch.randn(8, 8) @ torch.randn(8, 8)
    say("PASS", "torch runs on cpu", f"{torch.get_num_threads()} threads")
except Exception as e:
    say("FAIL", "torch runs on cpu", str(e)[:60])

for f in ("web/index.html", "web/m.html", "lab/train.py", "lab/papers.json"):
    say("PASS" if (ROOT / f).exists() else "FAIL", f"file {f}")

try:
    db = Path(os.environ.get("LAB_DB", ROOT / "data" / "lab.db"))
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE IF NOT EXISTS _pf(x)"); c.execute("INSERT INTO _pf VALUES(1)"); c.execute("DROP TABLE _pf"); c.commit(); c.close()
    say("PASS", "database writable", str(db.relative_to(ROOT)) if db.is_relative_to(ROOT) else str(db))
except Exception as e:
    say("FAIL", "database writable", str(e)[:60])

port = int(os.environ.get("PORT", "8000"))
s = socket.socket(); s.settimeout(0.5)
busy = s.connect_ex(("127.0.0.1", port)) == 0; s.close()
say("WARN" if busy else "PASS", f"port {port} free", "something is already listening: stop it before ./run.sh" if busy else "")

say("PASS" if shutil.which("cloudflared") else "WARN", "cloudflared installed", "needed only for ./run.sh --tunnel and phone control off-LAN")

key = os.environ.get("ELEVATE_TEAM_API_KEY", "")
say("PASS" if key else "WARN", "ELEVATE_TEAM_API_KEY set", "LLM features on" if key else "unset: the lab runs rule-based, no LLM")
url = os.environ.get("ELEVATE_URL", "https://elevate1-0.vercel.app/api/ai")
try:  # no key sent: a 401/403/405 still proves the gateway is reachable
    urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"}), timeout=8)
    say("PASS", "gateway reachable")
except urllib.error.HTTPError as e:
    say("PASS" if e.code in (400, 401, 403, 405, 422) else "WARN", "gateway reachable", f"HTTP {e.code}")
except Exception as e:
    say("WARN", "gateway reachable", f"{type(e).__name__}: venue network may block it; cached answers still work")

try:
    papers = (ROOT / "lab" / "papers.json").read_text()
    if '"verified": false' in papers:
        say("WARN", "paper claim verified", "lab/papers.json still has verified:false; the dashboard shows a warning")
except Exception:
    pass

print("\nREADY" if not bad else f"\nNOT READY: {bad} failure(s)")
sys.exit(1 if bad else 0)
