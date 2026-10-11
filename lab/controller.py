"""The meta-controller. One request becomes: plan -> staff (spawn with approval if a skill is missing) -> run (parallel,
memory-first) -> repair (fix memory) -> compare to the paper -> reward / retire / refine -> report."""
import asyncio, json, os, uuid
from . import agents as ag
from .agents import Team, compute_reward
from .bus import Bus
from .comparator import compare, summary_line, curve_facts
from .config import PAPERS, config_hash, validate
from .db import DB, now, jl
from .executor import Executor
from .llm import LLM
from .memory import Memory, PATCHABLE
from .planner import PRESETS, analyze, design_variants

AUTO_APPROVE = bool(os.environ.get("LAB_AUTO_APPROVE"))
APPROVAL_TIMEOUT_S = int(os.environ.get("LAB_APPROVAL_TIMEOUT", "900"))
MAX_PARALLEL = int(os.environ.get("LAB_PARALLEL", "3"))
RUN_BUDGET_S = int(os.environ.get("LAB_RUN_BUDGET_S", "3600"))

# What to hire when a needed skill has no active agent.
SPAWN_TEMPLATES = {
    "hyperparam_search": dict(name="HyperTuner", prompt="You expand a what-if axis into a small, informative hyperparameter sweep under a compute budget."),
    "paper_read": dict(name="PaperReader", prompt="You read a paper and extract its exact setup and claims."),
    "error_debug": dict(name="Debugger", prompt="You diagnose failed runs and propose minimal structured config patches."),
    "repro_compare": dict(name="Analyst", prompt="You explain where and why results differ from a paper, with evidence."),
}


class Lab:
    def __init__(self, db: DB | None = None):
        self.db = db or DB()
        self.bus = Bus(self.db)
        ag.seed(self.db)
        self.team = Team(self.db, self.bus)
        self.mem = Memory(self.db)
        self.llm = LLM(self.db)
        self.exe = Executor()
        self.sem = asyncio.Semaphore(MAX_PARALLEL)
        self.futures: dict[str, asyncio.Future] = {}
        self.tasks: set[asyncio.Task] = set()
        self.recover()

    def recover(self):
        """A restart kills the trainers, so anything still marked in-flight is a ghost: close it out honestly."""
        runs = self.db.all("SELECT id FROM runs WHERE status IN ('running','queued','waiting')")
        reqs = self.db.all("SELECT id FROM requests WHERE status IN ('planning','running')")
        apprs = self.db.all("SELECT id FROM approvals WHERE status='pending'")
        for r in runs:
            self.db.x("UPDATE runs SET status='failed', error_sig='interrupted by server restart', finished_at=? WHERE id=?", (now(), r["id"]))
        for r in reqs:
            self.db.x("UPDATE requests SET status='error', report='interrupted by server restart' WHERE id=?", (r["id"],))
        for a in apprs:
            self.db.x("UPDATE approvals SET status='rejected' WHERE id=?", (a["id"],))
        if runs or reqs or apprs:
            self.bus.emit("recovered", runs=len(runs), requests=len(reqs), approvals=len(apprs))

    # ---------------------------------------------------------------- requests
    def submit(self, text: str, options: dict | None = None) -> int:
        rid = self.db.x("INSERT INTO requests(text,status,created_at) VALUES(?,?,?)", (text, "planning", now()))
        self.bus.emit("request", request_id=rid, text=text)
        t = asyncio.create_task(self._process(rid, text, options or {}))
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return rid

    async def _process(self, rid, text, opt):
        try:
            await self._process_inner(rid, text, opt)
        except Exception as e:  # never die silently
            self.db.x("UPDATE requests SET status='error', report=? WHERE id=?", (f"{type(e).__name__}: {e}", rid))
            self.bus.emit("request_error", request_id=rid, error=f"{type(e).__name__}: {e}")

    async def _process_inner(self, rid, text, opt):
        bus, team = self.bus, self.team
        plan = await asyncio.to_thread(analyze, text, self.llm)
        preset = opt.get("preset", "fast")
        plan["preset"] = preset
        bus.emit("plan", request_id=rid, capabilities=plan["capabilities"], axes=plan["axes"], why=plan["why"], paper=plan["paper"], preset=preset)
        for l in self.mem.related_lessons(text):
            bus.emit("lesson_recalled", request_id=rid, text=l["text"], lesson_id=l["id"])

        # ---- staffing: every needed skill needs an active agent; a missing one is proposed to the researcher
        for cap in plan["capabilities"]:
            if team.active_with(cap):
                continue
            if cap == "code_execute" or cap not in SPAWN_TEMPLATES:
                continue
            tpl = SPAWN_TEMPLATES[cap]
            ok = await self._ask_approval(rid, "spawn", {"name": tpl["name"], "capability": cap,
                                                         "reason": f"No active agent has '{cap}'. Needed for: {text[:80]}"})
            if ok:
                team.spawn(tpl["name"], [cap], tpl["prompt"], origin="spawned")
            else:
                bus.emit("degraded", request_id=rid, message=f"Spawn of {tpl['name']} was rejected; continuing without '{cap}'.")

        # ---- paper_read
        paper_key = plan["paper"]
        paper = PAPERS[paper_key]
        reader = team.select("paper_read", rid)
        base = dict(paper["setup"])
        base.update(PRESETS.get(preset, {}))
        base["epochs"] = int(opt.get("epochs") or base["epochs"] if preset != "paper" else (opt.get("epochs") or 10000))
        if opt.get("seed") is not None:
            base["seed"] = int(opt["seed"])
        bus.emit("paper", request_id=rid, key=paper_key, title=paper["title"], authors=paper["authors"], year=paper["year"],
                 setup=paper["setup"], claim=paper["claim"], used=base, agent=reader["id"] if reader else None)
        if reader:
            self._reward_simple(reader, None, "paper_read", "paper card extracted")

        # ---- what-if design: Planner picks axes, HyperTuner expands them into a sweep
        planner = team.select("whatif_design", rid)
        tuner = team.select("hyperparam_search", rid)
        if planner:
            self._reward_simple(planner, None, "whatif_design", f"axes={plan['axes']}")
        variants = design_variants(plan["axes"], base, sweep=tuner is not None)
        if tuner:
            self._reward_simple(tuner, None, "hyperparam_search", f"{len(variants)} variants")
        bus.emit("variants", request_id=rid, variants=variants, by=tuner["id"] if tuner else None, planner=planner["id"] if planner else None)

        exps = [{"label": "baseline", "cfg": base, "question": f"Reproduce {paper['title'].split(':')[0]}"}]
        for v in variants:
            c = dict(base)
            c.update(v["patch"])
            exps.append({"label": v["label"], "cfg": c, "question": v["question"]})

        # ---- run
        self.db.x("UPDATE requests SET status='running', plan=? WHERE id=?", (json.dumps({**plan, "experiments": exps}), rid))
        jobs = [self._run_experiment(rid, e, opt.get("runner"), paper_key) for e in exps]
        results = await asyncio.gather(*jobs)
        if not any(r.get("run_id") for r in results):
            msg = "every experiment was rejected before running; check the config values (see the rejected_experiments log)"
            self.db.x("UPDATE requests SET status='error', report=? WHERE id=?", (msg, rid))
            bus.emit("request_error", request_id=rid, error=msg)
            return

        # ---- analysis
        analyst = team.select("repro_compare", rid)
        findings = await asyncio.to_thread(self._analyse, rid, paper_key, results, analyst)
        report = self._report(rid, text, paper_key, results, findings)
        await asyncio.to_thread(team.review_all, self.llm)
        self.db.x("UPDATE requests SET status='done', report=?, findings=? WHERE id=?", (report, json.dumps(findings), rid))
        bus.emit("report_ready", request_id=rid)

    # ---------------------------------------------------------------- approvals
    async def _ask_approval(self, rid, kind, payload) -> bool:
        aid = uuid.uuid4().hex[:8]
        self.db.x("INSERT INTO approvals(id,request_id,kind,payload,created_at) VALUES(?,?,?,?,?)", (aid, rid, kind, json.dumps(payload), now()))
        self.bus.emit("approval_requested", approval_id=aid, request_id=rid, kind=kind, **payload)
        if AUTO_APPROVE:
            self.decide(aid, True, "auto")
        fut = asyncio.get_event_loop().create_future()
        self.futures[aid] = fut
        if self.db.one("SELECT status FROM approvals WHERE id=?", (aid,))["status"] != "pending":
            return self.db.one("SELECT status FROM approvals WHERE id=?", (aid,))["status"] == "approved"
        try:
            return await asyncio.wait_for(fut, APPROVAL_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.decide(aid, False, "timeout")
            return False

    def decide(self, aid, approve: bool, via="dashboard"):
        row = self.db.one("SELECT * FROM approvals WHERE id=?", (aid,))
        if not row or row["status"] != "pending":
            return False
        st = "approved" if approve else "rejected"
        self.db.x("UPDATE approvals SET status=?, decided_at=?, decided_via=? WHERE id=?", (st, now(), via, aid))
        self.bus.emit("approval_decided", approval_id=aid, status=st, via=via, request_id=row["request_id"])
        f = self.futures.pop(aid, None)
        if f and not f.done():
            f.set_result(approve)
        return True

    # ---------------------------------------------------------------- one experiment
    async def _run_experiment(self, rid, exp, pin_runner, paper_key):
        bus, team = self.bus, self.team
        cfg_in = exp["cfg"]
        v, err = validate({**cfg_in, "label": exp["label"]})
        if err:
            bus.emit("config_rejected", request_id=rid, label=exp["label"], error=err)
            return {"label": exp["label"], "status": "rejected", "error": err}
        cfg = v.model_dump()
        h = config_hash(cfg)
        run_id = self.db.x("INSERT INTO runs(request_id,label,requested_hash,requested,status,started_at) VALUES(?,?,?,?,?,?)",
                           (rid, exp["label"], h, json.dumps(cfg), "queued", now()))
        hit = self.mem.find_run(h)
        if hit:
            self.db.x("UPDATE runs SET status='success', reused_from=?, result=?, effective=?, agent_id=?, finished_at=?, wall_s=0, cpu_s=0 WHERE id=?",
                      (hit["id"], hit["result"], hit["effective"], hit["agent_id"], now(), run_id))
            for m in self.db.all("SELECT * FROM metrics WHERE run_id=?", (hit["id"],)):
                self.db.x("INSERT INTO metrics VALUES(?,?,?,?,?,?)", (run_id, m["epoch"], m["train_acc"], m["test_acc"], m["train_loss"], m["test_loss"]))
            bus.emit("memory_hit", request_id=rid, run_id=run_id, label=exp["label"], source_run=hit["id"], hash=h,
                     message=f"Found in memory (run #{hit['id']}), reusing result. 0 CPU seconds spent.")
            return self._pack(run_id, exp, reused=True)

        agent = None
        if pin_runner and (a := team.get(pin_runner)) and a["status"] == "active" and "code_execute" in a["capabilities"]:
            agent = a
            team.pending[agent["id"]] = team.pending.get(agent["id"], 0) + 1
            bus.emit("runner_pinned", request_id=rid, agent=agent["id"], note="researcher override")
        else:
            agent = team.select("code_execute", rid)
        if not agent:
            self.db.x("UPDATE runs SET status='failed', error_sig='no agent with code_execute' WHERE id=?", (run_id,))
            bus.emit("run_failed", run_id=run_id, label=exp["label"], error="no active agent with code_execute")
            return self._pack(run_id, exp)

        eff = dict(cfg)
        pol = agent["policy"]
        if pol.get("device") == "paper":
            eff["device"] = PAPERS[paper_key].get("paper_device", "cuda")
        frac = float(pol.get("epoch_fraction", 1.0))
        truncated = frac < 1.0
        if truncated:
            eff["epochs"] = max(10, int(eff["epochs"] * frac))
        self.db.x("UPDATE runs SET agent_id=?, effective=?, status='waiting' WHERE id=?", (agent["id"], json.dumps(eff), run_id))
        bus.emit("run_queued", run_id=run_id, request_id=rid, label=exp["label"], agent=agent["id"], hash=h, config=eff, question=exp["question"])

        notes, fix_used, attempts, out = [], None, 0, None
        async with self.sem:
            t_start = now()
            while attempts < 3:
                attempts += 1
                self.db.x("UPDATE runs SET status='running', attempts=? WHERE id=?", (attempts, run_id))
                bus.emit("run_started", run_id=run_id, label=exp["label"], agent=agent["id"], attempt=attempts, device=eff["device"], epochs=eff["epochs"])

                def on_metric(m, _rid=run_id, _label=exp["label"]):
                    self.db.x("INSERT INTO metrics VALUES(?,?,?,?,?,?)", (_rid, m["epoch"], m["train_acc"], m["test_acc"], m["train_loss"], m["test_loss"]))
                    self.bus.emit("metric", run_id=_rid, label=_label, **m)

                def on_line(kind, text, _rid=run_id):
                    if kind == "optim":
                        self.bus.emit("optim_logged", run_id=_rid, text=text)

                self.db.x("DELETE FROM metrics WHERE run_id=?", (run_id,))
                out = await self.exe.run(run_id, eff, on_metric, on_line, budget_s=RUN_BUDGET_S)
                if out["status"] != "failed":
                    break
                sig = out["error_sig"]
                bus.emit("run_error", run_id=run_id, label=exp["label"], signature=sig, attempt=attempts)
                notes.append(sig)
                fix = self.mem.find_fix(sig)
                if fix:
                    bus.emit("fix_recalled", run_id=run_id, label=exp["label"], signature=sig, source_run=fix["run_id"],
                             patch=fix["fix"].get("patch"), message=f"Recalled fix from run #{fix['run_id']}: {fix['description']}")
                    patch, desc, from_memory = fix["fix"]["patch"], fix["description"], True
                else:
                    dbg = team.select("error_debug", rid)
                    patch, desc = await asyncio.to_thread(self._diagnose, sig, out["stderr_tail"], eff)
                    from_memory = False
                    bus.emit("fix_proposed", run_id=run_id, label=exp["label"], agent=dbg["id"] if dbg else None, signature=sig, patch=patch, description=desc)
                    if dbg:
                        team.release(dbg["id"])
                if not patch:
                    break
                eff.update(patch)
                fix_used = {"signature": sig, "patch": patch, "from_memory": from_memory, "description": desc}
                self.db.x("UPDATE runs SET effective=?, fix_used=? WHERE id=?", (json.dumps(eff), json.dumps(fix_used), run_id))

            wall = now() - t_start

        status = out["status"]
        if status == "success" and truncated:
            status = "insufficient"
            notes.append(f"insufficient: agent truncated run to {eff['epochs']} of {cfg['epochs']} epochs")
        res = out["result"] if out else {}
        self.db.x("UPDATE runs SET status=?, finished_at=?, wall_s=?, cpu_s=?, result=?, error_sig=?, attempts=?, log_tail=? WHERE id=?",
                  (status, now(), wall, out.get("cpu_s", 0), json.dumps(res), out.get("error_sig"), attempts, "\n".join(out.get("stderr_tail", [])[-8:]), run_id))
        team.release(agent["id"])

        if status == "killed":
            bus.emit("run_killed", run_id=run_id, label=exp["label"])
            return self._pack(run_id, exp)

        quality = {"success": 0.7 if fix_used else 1.0, "insufficient": 0.3}.get(status, 0.0)
        base_r = compute_reward(status, quality, wall, out.get("cpu_s", 0), budget_s=max(60, RUN_BUDGET_S / 4))
        team.record(agent["id"], run_id, "code_execute", base_r, "; ".join(notes[:2]))
        if fix_used and not fix_used["from_memory"]:  # only credit the Debugger for fixes it actually produced
            d = next(iter(team.active_with("error_debug")), None)
            if d:
                team.record(d["id"], run_id, "error_debug", compute_reward("success" if status == "success" else "failed", 1.0 if status == "success" else 0.0, 5, 0, 900), "")
        if fix_used and status == "success":
            if fix_used["from_memory"]:
                self.mem.bump_fix(fix_used["signature"])
            else:
                self.mem.save_fix(fix_used["signature"], fix_used["description"], {"patch": fix_used["patch"]}, run_id)
                bus.emit("fix_saved", run_id=run_id, signature=fix_used["signature"], patch=fix_used["patch"])
        bus.emit("run_done", run_id=run_id, label=exp["label"], status=status, agent=agent["id"], reward=base_r,
                 result=res, wall_s=round(wall, 1), attempts=attempts, fix=fix_used)
        new = await asyncio.to_thread(team.review, agent["id"], self.llm)
        return self._pack(run_id, exp, new_agent=new["id"] if new else None)

    def _diagnose(self, sig: str, stderr: list[str], eff: dict):
        """Debugger: map an error to a minimal structured patch. Patches are data, validated against PATCHABLE."""
        s = sig.lower()
        patch, desc = None, None
        if "cuda" in s and eff.get("device") == "cuda":
            patch, desc = {"device": "cpu"}, "CUDA unavailable on this machine: run on cpu"
        elif "mps" in s and eff.get("device") == "mps":
            patch, desc = {"device": "cpu"}, "MPS unavailable: run on cpu"
        elif self.llm.available:
            out = self.llm.ask('Return JSON only: {"patch": {...}, "description": "..."} using only keys ' + ",".join(sorted(PATCHABLE)),
                               f"Error: {sig}\nTail:\n" + "\n".join(stderr[-8:]) + f"\nConfig: {json.dumps(eff)}")
            try:
                j = json.loads(out[out.index("{"):out.rindex("}") + 1])
                if set(j["patch"]) <= PATCHABLE:
                    patch, desc = j["patch"], j.get("description", "LLM-proposed patch")
            except Exception:
                pass
        return patch, desc

    def _pack(self, run_id, exp, reused=False, new_agent=None):
        r = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        return {"run_id": run_id, "label": exp["label"], "status": r["status"], "cfg": jl(r["requested"]), "eff": jl(r["effective"]),
                "result": jl(r["result"], {}), "reused": reused, "question": exp.get("question"), "agent": r["agent_id"]}

    def _reward_simple(self, agent, run_id, cap, note):
        r = compute_reward("success", 1.0, 2, 0, 900)
        self.team.record(agent["id"], run_id, cap, r, note)
        self.team.release(agent["id"])

    # ---------------------------------------------------------------- analysis & report
    def _analyse(self, rid, paper_key, results, analyst):
        findings = []
        by = {r["label"]: r for r in results if r.get("run_id")}
        base = by.get("baseline")
        if base and base["status"] == "success":
            ms = self.db.all("SELECT * FROM metrics WHERE run_id=? ORDER BY epoch", (base["run_id"],))
            tv = base["result"].get("torch")
            cmp = compare(paper_key, base["eff"] or base["cfg"], ms, str(tv) if tv else None)
            cmp["summary"] = summary_line(cmp)
            if self.llm.available:
                t = self.llm.ask("In 2 plain sentences, explain to a researcher why this reproduction differs from the paper. Use only the evidence given.", json.dumps(cmp))
                if t:
                    cmp["plain"] = t.strip()
            findings.append({"kind": "repro", **cmp, "run_id": base["run_id"]})
            self.bus.emit("repro_compare", request_id=rid, run_id=base["run_id"], **{k: v for k, v in cmp.items() if k != "kind"})
            if cmp["we_observed"]["grok_epoch"]:
                self.mem.add_lesson(f"{cmp['paper_key']} setup (p={base['cfg']['p']}, wd={base['cfg']['weight_decay']}, train_frac={base['cfg']['train_frac']}) "
                                    f"memorised at epoch {cmp['we_observed']['memorise_epoch']} and groked at {cmp['we_observed']['grok_epoch']} on CPU.", "grokking reproduce", base["run_id"])
        for lab, r in by.items():
            if lab == "baseline" or r["status"] != "success":
                continue
            g = r["result"].get("grok_epoch") or 0
            bg = (base["result"].get("grok_epoch") or 0) if base else 0
            outcome = f"groked at epoch {g}" if g else f"did NOT grok within {r['result'].get('epochs_run', '?')} epochs (test acc {r['result'].get('final_test_acc')})"
            cmpr = f" (baseline: {'groked at ' + str(bg) if bg else 'no grok'})" if base else ""
            f = {"kind": "whatif", "label": lab, "question": r["question"], "outcome": outcome + cmpr, "run_id": r["run_id"]}
            findings.append(f)
            self.bus.emit("whatif_result", request_id=rid, **{k: v for k, v in f.items() if k != "kind"})
            self.mem.add_lesson(f"What-if {lab} ({r['question']}) {outcome}.", f"whatif {lab}", r["run_id"])
        if analyst:
            self._reward_simple(analyst, None, "repro_compare", f"{len(findings)} findings")
        return findings

    def _report(self, rid, text, paper_key, results, findings) -> str:
        p = PAPERS[paper_key]
        L = [f"# Research note: {text}", "", f"Paper: **{p['title']}** ({p['authors']}, {p['year']})", "",
             "## Experiments", "", "| run | label | status | grok epoch | final test acc | reused |", "|---|---|---|---|---|---|"]
        for r in results:
            if not r.get("run_id"):
                L.append(f"| – | {r['label']} | rejected | – | – | – |")
                continue
            res = r["result"] or {}
            L.append(f"| #{r['run_id']} | {r['label']} | {r['status']} | {res.get('grok_epoch') or '–'} | {res.get('final_test_acc', '–')} | {'yes' if r['reused'] else 'no'} |")
        L += ["", "## Findings", ""]
        for f in findings:
            if f["kind"] == "repro":
                L += [f"**Reproduction ({f['qualitative_match']})**: {f['summary']}", ""]
                if f.get("plain"):
                    L += [f"> {f['plain']}", ""]
            else:
                L.append(f"- **{f['question']}** → {f['outcome']}")
        if not any(f["kind"] == "repro" for f in findings):
            L.append("- No successful baseline run, so no reproduction claim is made.")
        L += ["", "_Generated by the research lab. Paper claims marked unverified come from lab/papers.json._"]
        return "\n".join(L)

    # ---------------------------------------------------------------- controls
    def kill(self, run_id): 
        ok = self.exe.kill(run_id)
        return ok

    def rate(self, run_id, rating):
        rows = self.team.rate(run_id, rating)
        self.db.x("UPDATE runs SET rating=? WHERE id=?", (rating, run_id))
        self.bus.emit("rated", run_id=run_id, rating=rating, agents=[r["agent_id"] for r in rows])
        for aid in {r["agent_id"] for r in rows}:
            self.team.review(aid, self.llm)

    def reset(self):
        for t in list(self.tasks):
            t.cancel()
        for rid in list(self.exe.procs):
            self.exe.kill(rid)
        for t in ("runs", "metrics", "events", "task_rewards", "lessons", "approvals", "requests", "fix_memory", "agents"):
            self.db.x(f"DELETE FROM {t}")
        ag.seed(self.db)
        self.team.pending.clear()
        self.bus.emit("reset")
