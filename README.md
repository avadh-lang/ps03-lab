# Evolving AI Research Lab (Elevate 1.0 · PS03)

A research environment where a team of agents reproduces a paper, investigates "what if?" questions, explains where
and why results differ from the paper, and learns across sessions. The team itself evolves: agents are spawned (with
human approval), scored, refined, merged and retired based on research outcome, compute cost and researcher feedback.

Name: **LabDex**. Agents are shown as creatures: spawn = catch, retire = faint, refine = evolve, merge = fuse. Type, level and
HP bar are computed from each agent's real capabilities, task count and score.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
./preflight.sh                       # readiness check: deps, torch, db, port, tunnel tool, key, gateway (no LLM tokens spent)
export ELEVATE_TEAM_API_KEY=...      # optional; set in the shell only, never in a file
./run.sh                             # dashboard on http://localhost:8000
./run.sh --tunnel                    # + public https URL and QR for phone control (needs `brew install cloudflared`)
```

The tunnel URL and phone token change on every restart. Scan the QR from the *current* dashboard, never an old one.
The tunnel uses HTTP/2 over TCP because QUIC (UDP) is blocked on many venue networks.

**LLM (optional).** With `ELEVATE_TEAM_API_KEY` the lab calls the Elevate gateway (OpenAI Chat Completions format, default
model `gpt-5-mini`, override with `LAB_MODEL`) for plan refinement, error diagnosis beyond the built-in rules, prompt
rewrites and plain-language explanations. Calls are metered per UTC day and stop at `LAB_TOKEN_CAP` (default 600,000 of the
800,000/day team allowance). Identical prompts are answered from a cache at zero cost; `LAB_REPLAY=1` serves only cached
answers. Without a key everything still works, deterministically and offline. The key is read from the environment only.

Tests: `.venv/bin/python -m pytest -q` (16 tests; one runs a real training subprocess end to end).

## The world (`/world`)

A walkable Phaser campus whose creatures are the lab's agents, driven by the same `/api/state` and `/ws` events as the
dashboard: new agents appear ("a wild X appeared"), running agents walk to the training field with a live accuracy bar,
retired agents faint, refined agents evolve. The baseline (reproduction) run opens a **battle**: the paper's claim is the boss, its HP drains as test accuracy climbs toward the 95% grok line, and your agent's HP is its epoch budget. Grok = victory; running out of epochs = your agent faints (every other run has a "Watch battle" button). Ask the lab, approve a spawn ("catch it") and kill a run from inside it.
Art and chiptune audio are code-generated (reused from the team's PokeStonks project), so there are no asset files.

```bash
cd world && npm install && npm run build     # bundles to web/world/world.js (committed output is not required)
./run.sh                                     # then open http://localhost:8000/world
```
Steel (CodeRunner-Careful) and bug (Debugger) creatures currently reuse other painters with their own colours.

## What is real

| Piece | Where | Notes |
|---|---|---|
| Grokking trainer | `lab/train.py` | Nanda-style 1-layer transformer, Nanda init, full-batch AdamW. Emits only `CONFIG/OPTIM/METRIC/RESULT` lines. |
| Executor | `lab/executor.py` | asyncio subprocess, killable (`killpg`), time budget, API keys scrubbed, explicit venv interpreter, `TQDM_DISABLE=1`. |
| Agents | `lab/agents.py` | An agent is a DB row. UCB1 per capability (the full table is emitted for every choice), reward formula, spawn / refine / merge / retire. |
| Memory | `lab/memory.py` | Exact run dedup (sha256 of config), fix memory (error signature to structured patch), lessons (keyword retrieval). |
| ReproComparator | `lab/comparator.py` | Every cause is derived from a concrete difference (config diff, epoch cap, curve shape, environment). |
| Controller | `lab/controller.py` | plan, staff, approve, run in parallel, repair, compare, reward, report. |
| Party + task graph | `web/index.html`, `lab/taskgraph.py` | Creature cards per agent; click one for a graph of lineage, skills, tasks and steps (error, fix, result) built from real rows, live while runs are going. |
| Dashboard / phone | `web/index.html`, `web/m.html` | Live curves, decision feed with UCB tables, approvals, kill, rate, memory, research note. |
| LLM gateway | `lab/llm.py` | Metered, capped, cached, retried; runs off the event loop so a slow call cannot freeze the UI. |
| Restart safety | `Lab.recover` | After a restart, runs/requests/approvals that were in flight are closed out instead of left as ghosts. |

## How the PS requirements map

| PS03 asks | Implementation |
|---|---|
| Reproduce a paper's key result on modest compute | Grokking, laptop CPU, ~50 ms/epoch at 2 threads |
| "What if?" investigated without hand-designing runs | Planner picks axes, HyperTuner expands them into a sweep, all run in parallel |
| Explain where and why results differ | ReproComparator, evidence per cause, strongest first |
| Meta-controller; spawn / adapt agents | A missing skill pauses the request and asks for approval (dashboard or phone) |
| Reward loop, refine / merge / retire | `compute_reward`, `Team.review`, `Team.refine`, `Team.merge` |
| Learn across sessions | Run memory, fix memory, lessons, agent scores persisted in SQLite |
| Control remotely; approve / reject / redirect | Phone page: approve, reject, kill. Token required off-localhost. |
| Understandable interface | Decision feed explains *why* (UCB table, plan rationale), creature party, per-agent task graph, per-run provenance |

## Honest limits

* **The paper claim in `lab/papers.json` is unverified** (`"verified": false`). The grok epoch was copied from the team's
  notes. Check it against the paper and edit the file. The dashboard shows a warning until you set `"verified": true`.
* Only grokking (Nanda setup, Power setup as a comparison) is implemented. Adding a paper means adding a trainer that
  speaks the same METRIC/RESULT protocol and an entry in `papers.json`.
* `paper_read` uses the local registry in `papers.json`. It does not fetch or parse arbitrary PDFs.
* Local execution only. Kaggle/Colab/remote executors are not implemented (the executor is the single place to add them).
* Lessons use keyword retrieval, not embeddings. Meaning-based search is deliberately not used for dedup: it cannot tell
  `weight_decay=0` from `1.0`.
* **The LLM path has only had a tiny test call and no full request.** The rule-based path is what the verified runs used.
  The gateway is shared (800,000 tokens/day for the team) and its latency varies; calls are retried and fall back to rules.
* The paper-size setup (p=113, 30% data) has not been run to completion with the corrected init, so there is no measured
  grok epoch for it. Only the `Fast` preset has.
* Killing works only for runs that have started; a queued run must start first.
* No LangGraph: the approval gate is an asyncio future, and spawning is a row insert either way.
* `Fast` scale (p=59, 50% data) is not the paper's setup. The ReproComparator reports that difference as a cause.

## Findings from building it

* On a laptop CPU the paper setup runs at roughly 50 ms/epoch (2 threads; the lab now defaults to 1 thread per run at low priority so the UI stays responsive), about 8 min per 10k epochs, much better than
  the 53 min to 2 h estimated in planning. Metrics are bit-identical at 1, 2 and 4 threads.
* During memorisation, test accuracy sits near the train fraction (not near chance). (a,b) and (b,a) have the same answer, so
  a memorising model gets the mirrored pairs of its training set right. The plateau is a feature of the task, not a bug.
* Initialisation matters a lot. PyTorch defaults with N(0,1) embeddings stall far longer than Nanda's N(0, 1/d) init.
* Running 3-4 trainers with several threads each starved the laptop (load average above 10) and made the dashboard look
  frozen. Trainers now run at `nice 15` with 1 thread, and reset stops them instead of orphaning them.
* Real run on the Fast preset (p=59, 50% data): baseline memorised at epoch 100 and grokked at 1000 (99.6% test accuracy);
  weight decay 0 and 0.1 did not grok within 3000 epochs. A pinned CodeRunner hit a real CUDA error, the Debugger patched it
  to `cpu`, and later runs reused the fix from memory.
