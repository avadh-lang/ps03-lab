"""Per-agent task graph, built from real rows only (agents, task_rewards, runs, fix_used).

Columns left to right:  lineage  ->  agent  ->  modules (its capabilities)  ->  tasks  ->  steps
A code-runner task expands into steps: [memory hit] | [error -> fix] -> result.
Nothing here is invented: every node maps to a row, and running work appears before it is rewarded.
"""
import json

MAX_TASKS_PER_MODULE = 6
MAX_RUN_TASKS = 10


def _j(x, d=None):
    try:
        return json.loads(x) if x else d
    except Exception:
        return d


def _status(run_status):
    return {"success": "done", "insufficient": "warn", "failed": "failed", "killed": "killed", "timeout": "failed",
            "running": "running", "queued": "waiting", "waiting": "waiting", "memory": "done"}.get(run_status, "waiting")


def agent_graph(db, team, aid: str) -> dict | None:
    a = team.get(aid)
    if not a:
        return None
    nodes, edges = [], []

    def node(id_, kind, label, sub="", status="done", **extra):
        nodes.append({"id": id_, "kind": kind, "label": label, "sub": sub, "status": status, **extra})
        return id_

    def edge(a_, b_):
        edges.append([a_, b_])

    root = node("agent", "agent", a["id"], f"v{a['version']} · {a['origin']} · score {a['avg_reward']:.2f}",
                "fainted" if a["status"] == "retired" else "done")

    for k, pid in enumerate([a.get("parent_id"), a.get("parent2_id")]):
        if pid:
            n = node(f"parent{k}", "parent", pid, "fused from" if a["origin"] == "merged" else "evolved from", "done")
            edge(n, root)

    if a["status"] == "retired":
        n = node("fainted", "step", "fainted", (a.get("retirement_reason") or "")[:90], "failed")
        edge(root, n)

    rewards = db.all("SELECT * FROM task_rewards WHERE agent_id=? ORDER BY id DESC LIMIT 80", (aid,))
    by_run_reward = {r["run_id"]: r for r in rewards if r["run_id"] and r["capability"] == "code_execute"}
    runs = db.all("SELECT * FROM runs WHERE agent_id=? ORDER BY id DESC LIMIT ?", (aid, MAX_RUN_TASKS))

    for cap in a["capabilities"]:
        caprows = [r for r in rewards if r["capability"] == cap]
        n_items = len(runs) if cap == "code_execute" else len(caprows)
        unit = "run" if cap == "code_execute" else "reward"
        mod = node(f"m:{cap}", "module", cap, f"{n_items} {unit}{'s' if n_items != 1 else ''}" if n_items else "no tasks yet",
                   "done" if n_items or cap != "code_execute" else "waiting")
        edge(root, mod)

        if cap == "code_execute":
            for r in reversed(runs):
                rw = by_run_reward.get(r["id"])
                t = node(f"t{r['id']}", "task", f"run #{r['id']} {r['label']}",
                         f"{r['status']}" + (f" · reward {rw['reward']:.2f}" if rw else ""), _status(r["status"]), run_id=r["id"])
                edge(mod, t)
                prev = t
                fx = _j(r["fix_used"])
                res = _j(r["result"], {}) or {}
                if r["reused_from"]:
                    s = node(f"s{r['id']}m", "step", "memory hit", f"reused run #{r['reused_from']}, 0 CPU", "done")
                    edge(prev, s)
                    prev = s
                if fx:
                    e = node(f"s{r['id']}e", "step", "error", (fx.get("signature") or "")[:60], "failed")
                    f = node(f"s{r['id']}f", "step", "fix " + json.dumps(fx.get("patch") or {}), "recalled from memory" if fx.get("from_memory") else "proposed by Debugger", "done")
                    edge(prev, e); edge(e, f)
                    prev = f
                if r["status"] in ("running", "waiting", "queued"):
                    s = node(f"s{r['id']}r", "step", "training…", "", "running")
                else:
                    gk = res.get("grok_epoch")
                    s = node(f"s{r['id']}r", "step", "result", (f"grokked @ {gk}" if gk else f"no grok · {r['status']}"), _status(r["status"]))
                edge(prev, s)
        else:
            for r in reversed(caprows[:MAX_TASKS_PER_MODULE]):
                t = node(f"r{r['id']}", "task", (r["note"] or cap)[:34], f"reward {r['reward']:.2f}" + (f" · rated {r['rating']}" if r["rating"] is not None else ""),
                         "done" if r["reward"] >= 0.3 else "failed", run_id=r["run_id"])
                edge(mod, t)
    return {"agent": aid, "status": a["status"], "nodes": nodes, "edges": edges}
