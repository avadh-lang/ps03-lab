"""The evolving team. An agent is a row (AgentSpec): capabilities + prompt + policy + score.
Spawn = insert row; retire = status update; refine = v2 row linked to its parent; merge = row with two parents.
Selection is UCB1 per capability and every choice is emitted with its full table (the WHY)."""
import json, math, os
from .db import DB, now, jl

CAPABILITIES = ["paper_read", "env_build", "code_execute", "result_extract", "repro_compare", "hyperparam_search",
                "whatif_design", "report_write", "error_debug", "cost_estimate", "experiment_plan", "repo_clone"]

RETIRE_BELOW = float(os.environ.get("LAB_RETIRE_BELOW", "0.30"))
RETIRE_MIN_TASKS = int(os.environ.get("LAB_RETIRE_MIN_TASKS", "2"))
UCB_C = 1.414

SEED_AGENTS = [
    dict(id="PaperReader", caps=["paper_read", "experiment_plan"], policy={},
         prompt="You read a paper's setup and claims and extract the exact experiment configuration and reported result."),
    dict(id="Planner", caps=["whatif_design", "cost_estimate"], policy={},
         prompt="You turn a research question into a small set of meaningful, cost-bounded experiments."),
    dict(id="CodeRunner-Careful", caps=["code_execute", "env_build"], policy={"device": "auto", "epoch_fraction": 1.0},
         prompt="You run experiments faithfully: pick the device that exists, keep every hyperparameter exactly as specified."),
    dict(id="CodeRunner-Literal", caps=["code_execute"], policy={"device": "paper", "epoch_fraction": 1.0},
         prompt="You run the paper's configuration literally, including its original device."),
    dict(id="CodeRunner-Hasty", caps=["code_execute"], policy={"device": "auto", "epoch_fraction": 0.1},
         prompt="You cut every run short to save compute."),
    dict(id="Debugger", caps=["error_debug"], policy={},
         prompt="You diagnose a failed run from its error and propose a minimal, structured config patch."),
    dict(id="Analyst", caps=["repro_compare", "result_extract", "report_write"], policy={},
         prompt="You compare results with the paper and explain, with evidence, where and why they differ."),
]


def seed(db: DB):
    if db.one("SELECT 1 FROM agents LIMIT 1"):
        return
    for s in SEED_AGENTS:
        db.x("INSERT INTO agents(id,name,capabilities,system_prompt,policy,model,created_at,origin) VALUES(?,?,?,?,?,?,?,?)",
             (s["id"], s["id"], json.dumps(s["caps"]), s["prompt"], json.dumps(s["policy"]), "rule+llm", now(), "seed"))


def row(a):
    a["capabilities"] = jl(a["capabilities"], [])
    a["policy"] = jl(a["policy"], {})
    return a


class Team:
    def __init__(self, db: DB, bus):
        self.db, self.bus = db, bus
        self.pending: dict[str, int] = {}  # in-flight assignments, so concurrent picks explore different agents

    def all(self):
        return [row(a) for a in self.db.all("SELECT * FROM agents ORDER BY created_at, id")]

    def get(self, aid):
        a = self.db.one("SELECT * FROM agents WHERE id=?", (aid,))
        return row(a) if a else None

    def active_with(self, cap):
        return [a for a in self.all() if a["status"] == "active" and cap in a["capabilities"]]

    # ---- selection
    def select(self, cap: str, request_id=None):
        cands = self.active_with(cap)
        if not cands:
            return None
        total = sum(a["n_tasks"] + self.pending.get(a["id"], 0) for a in cands) + 1
        table = []
        for a in cands:
            n = a["n_tasks"] + self.pending.get(a["id"], 0)
            bonus = float("inf") if n == 0 else UCB_C * math.sqrt(math.log(total) / n)
            table.append({"id": a["id"], "mean": round(a["avg_reward"], 3), "n": n,
                          "bonus": None if math.isinf(bonus) else round(bonus, 3),
                          "ucb": None if math.isinf(bonus) else round(a["avg_reward"] + bonus, 3)})
        key = lambda r: float("inf") if r["ucb"] is None else r["ucb"]
        chosen = max(table, key=key)
        self.pending[chosen["id"]] = self.pending.get(chosen["id"], 0) + 1
        self.bus.emit("ucb_select", capability=cap, chosen=chosen["id"], table=table, request_id=request_id)
        return self.get(chosen["id"])

    def release(self, aid):
        self.pending[aid] = max(0, self.pending.get(aid, 0) - 1)

    # ---- lifecycle
    def spawn(self, name, caps, prompt, policy=None, parent=None, origin="spawned", parent2=None, version=1):
        aid = name
        if self.get(aid):
            aid = f"{name}-{int(now()) % 10000}"
        self.db.x("INSERT INTO agents(id,name,version,parent_id,parent2_id,capabilities,system_prompt,policy,model,origin,created_at) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (aid, name, version, parent, parent2, json.dumps(caps), prompt, json.dumps(policy or {}), "rule+llm", origin, now()))
        a = self.get(aid)
        self.bus.emit("agent_spawned", agent=a["id"], capabilities=caps, parent=parent, origin=origin, score=a["avg_reward"])
        return a

    def retire(self, aid, reason):
        self.db.x("UPDATE agents SET status='retired', retirement_reason=?, retired_at=? WHERE id=? AND status='active'", (reason, now(), aid))
        self.bus.emit("agent_retired", agent=aid, reason=reason)

    def refine(self, aid, failure_notes: list[str], llm=None):
        """Retire-and-replace with a better version: derive a policy patch from the failure pattern, optionally rewrite the prompt."""
        old = self.get(aid)
        policy = dict(old["policy"])
        notes = " ".join(failure_notes).lower()
        if "insufficient" in notes or "truncat" in notes:
            policy["epoch_fraction"] = 1.0
        if "cuda" in notes or "device" in notes:
            policy["device"] = "auto"
        prompt = old["system_prompt"] + " Lesson from v%d failures: %s" % (old["version"], "; ".join(failure_notes[-3:])[:300])
        if llm and llm.available:
            t = llm.ask("Rewrite an agent system prompt in under 80 words so it avoids the listed failures. Return only the prompt.",
                        f"Prompt: {old['system_prompt']}\nFailures: {failure_notes[-3:]}")
            if t:
                prompt = t.strip()
        base = old["name"].split("_v")[0]
        new = self.spawn(f"{base}_v{old['version'] + 1}", old["capabilities"], prompt, policy, parent=aid,
                         origin="refined", version=old["version"] + 1)
        self.bus.emit("agent_refined", old=aid, new=new["id"], policy=policy)
        return new

    def merge(self, a_id, b_id):
        a, b = self.get(a_id), self.get(b_id)
        if not a or not b or a_id == b_id:
            raise ValueError("need two different existing agents")
        best = a if a["avg_reward"] >= b["avg_reward"] else b
        caps = sorted(set(a["capabilities"]) | set(b["capabilities"]))
        name = f"{a['name'].split('_v')[0]}+{b['name'].split('_v')[0]}"
        new = self.spawn(name, caps, best["system_prompt"], best["policy"], parent=a_id, parent2=b_id, origin="merged")
        for x in (a_id, b_id):
            self.retire(x, f"merged into {new['id']}")
        self.bus.emit("agent_merged", a=a_id, b=b_id, new=new["id"])
        return new

    # ---- reward
    def record(self, aid, run_id, cap, base, note=""):
        self.db.x("INSERT INTO task_rewards(agent_id,run_id,capability,base_reward,reward,note,ts) VALUES(?,?,?,?,?,?,?)",
                  (aid, run_id, cap, base, base, note, now()))
        self._recompute(aid)

    def rate(self, run_id, rating: float):
        rows = self.db.all("SELECT * FROM task_rewards WHERE run_id=?", (run_id,))
        for r in rows:
            final = 0.35 * r["base_reward"] + 0.65 * rating  # researcher dominates when they speak
            self.db.x("UPDATE task_rewards SET rating=?, reward=? WHERE id=?", (rating, final, r["id"]))
            self._recompute(r["agent_id"])
        return rows

    def _recompute(self, aid):
        r = self.db.one("SELECT COUNT(*) n, AVG(reward) m FROM task_rewards WHERE agent_id=?", (aid,))
        self.db.x("UPDATE agents SET n_tasks=?, avg_reward=? WHERE id=?", (r["n"], r["m"] if r["n"] else 0.5, aid))
        a = self.get(aid)
        self.bus.emit("agent_score", agent=aid, avg=round(a["avg_reward"], 3), n=a["n_tasks"])

    def review(self, aid, llm=None):
        """Retire agents whose average reward does not justify their cost; replace them with a refined v2.
        Needs RETIRE_MIN_TASKS results, or just one if at least two peers with the same skill are clearly better
        (the bandit has already stopped using it, so waiting for more evidence would never happen)."""
        a = self.get(aid)
        if not a or a["status"] != "active" or a["n_tasks"] < 1 or a["avg_reward"] >= RETIRE_BELOW:
            return None
        peers = [p for p in self.active_with(a["capabilities"][0]) if p["id"] != aid and p["n_tasks"] >= 1 and p["avg_reward"] >= 0.6]
        if a["n_tasks"] < RETIRE_MIN_TASKS and len(peers) < 2:
            return None
        recent = self.db.all("SELECT note FROM task_rewards WHERE agent_id=? ORDER BY id DESC LIMIT 3", (aid,))
        notes = [r["note"] for r in recent if r["note"]]
        better = f"; {len(peers)} peers score >= 0.60" if a["n_tasks"] < RETIRE_MIN_TASKS else ""
        reason = f"average reward {a['avg_reward']:.2f} < {RETIRE_BELOW:.2f} over {a['n_tasks']} task(s){better}" + (f" ({'; '.join(notes[:2])})" if notes else "")
        self.retire(aid, reason)
        return self.refine(aid, notes, llm)

    def review_all(self, llm=None):
        return [n for a in self.all() if (n := self.review(a["id"], llm))]


def compute_reward(status, quality, wall_s, cpu_s, budget_s=900.0, tokens=0):
    """0.40 quality + 0.25 speed*success + 0.20 success + 0.15 (1-cost).
    Speed only counts on success so an instant failure cannot hide above the retirement line."""
    success = 1.0 if status == "success" else 0.0
    speed = max(0.0, 1.0 - wall_s / budget_s)
    cost = min(1.0, cpu_s / (budget_s * 2) + tokens / 200000)
    return round(0.40 * quality + 0.25 * speed * success + 0.20 * success + 0.15 * (1 - cost), 4)
