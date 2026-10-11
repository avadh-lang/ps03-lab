"""FastAPI app: REST + WebSocket (heartbeat, replay-on-reconnect) + phone approval route.
Anything that does not come from localhost needs the bearer token (the tunnel makes the app public)."""
import asyncio, io, json, os, secrets, socket
from pathlib import Path
from .taskgraph import agent_graph
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .controller import Lab
from .db import ROOT, jl

TOKEN = os.environ.get("LAB_TOKEN") or secrets.token_urlsafe(9)
WEB = ROOT / "web"
app = FastAPI(title="Evolving Research Lab")
lab: Lab | None = None


@app.on_event("startup")
async def _start():
    global lab
    lab = Lab()
    print(f"\n  Mobile token: {TOKEN}\n  Phone URL:    {public_base()}/m?token={TOKEN}\n", flush=True)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def public_base():
    f = ROOT / "data" / "tunnel_url.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip().rstrip("/")
    return f"http://{lan_ip()}:{os.environ.get('PORT', '8000')}"


def is_remote(req) -> bool:
    h = req.headers
    if any(k in h for k in ("cf-connecting-ip", "x-forwarded-for", "x-forwarded-host")):
        return True
    return (req.client.host if req.client else "") not in ("127.0.0.1", "::1", "localhost")


def auth(req: Request):
    if not is_remote(req):
        return
    tok = req.headers.get("authorization", "").removeprefix("Bearer ").strip() or req.query_params.get("token") or req.cookies.get("lab_token")
    if tok != TOKEN:
        raise HTTPException(401, "token required")


# ---------------------------------------------------------------- pages
@app.get("/")
def index(request: Request):
    auth(request)
    r = FileResponse(WEB / "index.html")
    if request.query_params.get("token") == TOKEN:
        r.set_cookie("lab_token", TOKEN, httponly=True, samesite="lax")
    return r


@app.get("/world")
def world(request: Request):
    auth(request)
    r = FileResponse(WEB / "world" / "index.html")
    if request.query_params.get("token") == TOKEN:
        r.set_cookie("lab_token", TOKEN, httponly=True, samesite="lax")
    return r


@app.get("/world-assets/{name}")
def world_asset(name: str):
    f = (WEB / "world" / name).resolve()
    if f.parent != (WEB / "world").resolve() or not f.is_file() or name == "index.html":
        raise HTTPException(404, "not found")
    return FileResponse(f)


@app.get("/m")
def mobile(request: Request):
    auth(request)
    r = FileResponse(WEB / "m.html")
    r.set_cookie("lab_token", TOKEN, httponly=True, samesite="lax")
    return r


@app.get("/api/qr.svg")
def qr(request: Request):
    auth(request)
    import qrcode, qrcode.image.svg
    img = qrcode.make(f"{public_base()}/m?token={TOKEN}", image_factory=qrcode.image.svg.SvgPathImage, box_size=8)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml")


# ---------------------------------------------------------------- state
def run_row(r):
    for k in ("requested", "effective", "result", "fix_used"):
        r[k] = jl(r.get(k))
    r.pop("log_tail", None) if False else None
    return r


@app.get("/api/state", dependencies=[Depends(auth)])
def state():
    db = lab.db
    last = db.one("SELECT COALESCE(MAX(id),0) m FROM events")["m"]
    reqs = db.all("SELECT id,text,status,created_at,plan,findings FROM requests ORDER BY id DESC LIMIT 20")
    for r in reqs:
        r["plan"], r["findings"] = jl(r["plan"]), jl(r["findings"], [])
    return {
        "last_event_id": last,
        "agents": lab.team.all(),
        "runs": [run_row(r) for r in db.all("SELECT * FROM runs ORDER BY id DESC LIMIT 60")],
        "requests": reqs,
        "approvals": [{**a, "payload": jl(a["payload"])} for a in db.all("SELECT * FROM approvals ORDER BY created_at DESC LIMIT 30")],
        "fixes": [{**f, "fix": jl(f["fix"])} for f in db.all("SELECT * FROM fix_memory ORDER BY created_at DESC")],
        "lessons": db.all("SELECT * FROM lessons ORDER BY id DESC LIMIT 30"),
        "llm": lab.llm.available,
        "llm_usage": lab.llm.usage(),
        "mobile": {"url": f"{public_base()}/m", "tunnel": (ROOT / "data" / "tunnel_url.txt").exists()},
        "events": lab.bus.since(max(0, last - 150)),
    }


@app.get("/api/events", dependencies=[Depends(auth)])
def events(since: int = 0):
    return lab.bus.since(since)


@app.get("/api/agents/{aid}/graph", dependencies=[Depends(auth)])
def agent_task_graph(aid: str):
    g = agent_graph(lab.db, lab.team, aid)
    if not g:
        raise HTTPException(404, "no such agent")
    return g


@app.get("/api/runs/{rid}/metrics", dependencies=[Depends(auth)])
def metrics(rid: int):
    return lab.db.all("SELECT epoch,train_acc,test_acc,train_loss,test_loss FROM metrics WHERE run_id=? ORDER BY epoch", (rid,))


@app.get("/api/requests/{rid}/report", dependencies=[Depends(auth)])
def report(rid: int):
    r = lab.db.one("SELECT report FROM requests WHERE id=?", (rid,))
    if not r:
        raise HTTPException(404)
    return PlainTextResponse(r["report"] or "(not ready)", media_type="text/markdown")


# ---------------------------------------------------------------- actions
class Req(BaseModel):
    text: str
    preset: str = "fast"
    epochs: int | None = None
    seed: int | None = None
    runner: str | None = None


@app.post("/api/requests", dependencies=[Depends(auth)])
async def new_request(body: Req):
    if not body.text.strip():
        raise HTTPException(400, "empty request")
    if len(body.text) > 2000:
        raise HTTPException(400, "request text is limited to 2000 characters")
    if body.preset not in ("fast", "paper"):
        raise HTTPException(400, "preset must be 'fast' or 'paper'")
    if body.seed is not None and not 0 <= body.seed <= 2**31 - 1:
        raise HTTPException(400, "seed must be between 0 and 2147483647")
    if body.epochs is not None and not 1 <= body.epochs <= 200_000:
        raise HTTPException(400, "epoch cap must be between 1 and 200000")
    rid = lab.submit(body.text.strip(), body.model_dump(exclude={"text"}))
    return {"request_id": rid}


class Decision(BaseModel):
    approve: bool


@app.post("/api/approvals/{aid}/decide", dependencies=[Depends(auth)])
async def decide(aid: str, body: Decision, request: Request):
    via = "phone" if is_remote(request) or request.headers.get("x-via") == "phone" else "dashboard"
    if not lab.decide(aid, body.approve, via):
        raise HTTPException(409, "already decided or unknown")
    return {"ok": True}


@app.post("/api/runs/{rid}/kill", dependencies=[Depends(auth)])
async def kill(rid: int):
    if not lab.kill(rid):
        raise HTTPException(409, "run is not active")
    return {"ok": True}


class Rating(BaseModel):
    rating: float


@app.post("/api/runs/{rid}/rate", dependencies=[Depends(auth)])
async def rate(rid: int, body: Rating):
    if body.rating not in (0, 0.5, 1):
        raise HTTPException(400, "rating must be 0, 0.5 or 1")
    await asyncio.to_thread(lab.rate, rid, body.rating)
    return {"ok": True}


class Merge(BaseModel):
    a: str
    b: str


@app.post("/api/agents/merge", dependencies=[Depends(auth)])
async def merge(body: Merge):
    try:
        return lab.team.merge(body.a, body.b)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/agents/{aid}/retire", dependencies=[Depends(auth)])
async def retire(aid: str):
    if not lab.team.get(aid):
        raise HTTPException(404)
    lab.team.retire(aid, "retired manually by researcher")
    return {"ok": True}


@app.post("/api/reset", dependencies=[Depends(auth)])
async def reset():
    lab.reset()
    return {"ok": True}


# ---------------------------------------------------------------- live feed
@app.websocket("/ws")
async def ws(sock: WebSocket):
    if is_remote(sock) and (sock.query_params.get("token") != TOKEN and sock.cookies.get("lab_token") != TOKEN):
        await sock.close(code=4401)
        return
    await sock.accept()
    since = int(sock.query_params.get("since", 0))
    q = lab.bus.subscribe()
    try:
        for ev in lab.bus.since(since):  # replay everything missed while disconnected
            await sock.send_json(ev)
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=10)
                await sock.send_json(ev)
            except asyncio.TimeoutError:
                await sock.send_json({"type": "heartbeat"})  # keeps proxies from dropping a quiet socket
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        lab.bus.unsubscribe(q)
