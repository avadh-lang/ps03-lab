"""Runs train.py as a killable subprocess. Parses only structured METRIC/RESULT lines."""
import asyncio, json, os, re, signal, sys
from collections import deque
from pathlib import Path

TRAIN = Path(__file__).parent / "train.py"
PY = os.environ.get("LAB_PYTHON", sys.executable)  # explicit interpreter, never PATH's `python`
THREADS = os.environ.get("LAB_THREADS", "1")
SECRET_KEYS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def clean_env():
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in SECRET_KEYS)}
    env.update(OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS,
               VECLIB_MAXIMUM_THREADS=THREADS, TQDM_DISABLE="1", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    return env


def parse_kv(line: str) -> dict:
    out = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                out[k] = v
    return out


def error_signature(stderr_tail: list[str]) -> str:
    for ln in reversed(stderr_tail):
        if re.match(r"^\s*[\w.]*(Error|Exception)\b", ln):
            return ln.strip()[:200]
    for ln in reversed(stderr_tail):
        if ln.strip():
            return ln.strip()[:200]
    return "unknown error"


class Executor:
    def __init__(self):
        self.procs: dict[int, asyncio.subprocess.Process] = {}
        self.killed: set[int] = set()

    async def run(self, run_id: int, cfg: dict, on_metric, on_line=None, budget_s=1800) -> dict:
        proc = await asyncio.create_subprocess_exec(
            PY, str(TRAIN), "--config", json.dumps(cfg),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, env=clean_env(),
            preexec_fn=lambda: os.nice(15))  # trainers yield to the server and browser
        self.procs[run_id] = proc
        err: deque[str] = deque(maxlen=40)
        result, optim, last_metric = {}, None, None

        async def read_out():
            nonlocal result, optim, last_metric
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if line.startswith("METRIC "):
                    last_metric = parse_kv(line)
                    on_metric(last_metric)
                elif line.startswith("RESULT "):
                    result = parse_kv(line)
                elif line.startswith("OPTIM "):
                    optim = line[6:]
                    if on_line:
                        on_line("optim", optim)

        async def read_err():
            async for raw in proc.stderr:
                err.append(raw.decode(errors="replace").rstrip())

        t0 = asyncio.get_event_loop().time()
        status = "success"
        try:
            await asyncio.wait_for(asyncio.gather(read_out(), read_err(), proc.wait()), timeout=budget_s)
        except asyncio.TimeoutError:
            status = "timeout"
            self._kill(proc)
            await proc.wait()
        wall = asyncio.get_event_loop().time() - t0
        self.procs.pop(run_id, None)
        if run_id in self.killed:
            self.killed.discard(run_id)
            status = "killed"
        elif status != "timeout" and (proc.returncode != 0 or not result):
            status = "failed"
        return {"status": status, "exit_code": proc.returncode, "result": result, "optim": optim,
                "last_metric": last_metric, "wall_s": wall, "cpu_s": result.get("cpu_s", 0.0),
                "stderr_tail": list(err), "error_sig": error_signature(list(err)) if status == "failed" else None}

    def kill(self, run_id: int) -> bool:
        proc = self.procs.get(run_id)
        if not proc:
            return False
        self.killed.add(run_id)
        self._kill(proc)
        return True

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
