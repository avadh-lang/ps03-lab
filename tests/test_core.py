import asyncio, json, os, tempfile
import pytest

os.environ["LAB_DB"] = tempfile.mktemp(suffix=".db")
os.environ["LAB_AUTO_APPROVE"] = "1"

from lab.agents import Team, compute_reward, seed
from lab.bus import Bus
from lab.comparator import compare, curve_facts
from lab.config import config_hash, validate
from lab.db import DB
from lab.executor import error_signature, parse_kv
from lab.memory import Memory
from lab.planner import analyze, design_variants


@pytest.fixture
def db():
    d = DB(tempfile.mktemp(suffix=".db"))
    seed(d)
    return d


def test_hash_ignores_presentation_fields_but_not_science():
    a = config_hash({"weight_decay": 1.0, "label": "x", "eval_every": 10})
    assert a == config_hash({"weight_decay": 1.0, "label": "y", "eval_every": 99})
    assert a != config_hash({"weight_decay": 0.0})  # the exact case semantic search gets wrong


def test_invalid_config_is_rejected_and_logged():
    cfg, err = validate({"weight_decay": "high"})
    assert cfg is None and "weight_decay" in err


def test_fast_failure_cannot_hide_above_retirement_line():
    assert compute_reward("failed", 0.0, 1, 1) < 0.30  # speed term only counts on success
    assert compute_reward("success", 1.0, 60, 60) > 0.8


def test_ucb_explores_untried_then_exploits(db):
    team = Team(db, Bus(db))
    seen = {team.select("code_execute")["id"] for _ in range(3)}
    assert seen == {"CodeRunner-Careful", "CodeRunner-Literal", "CodeRunner-Hasty"}  # pending counts force exploration
    for k in list(team.pending):
        team.pending[k] = 0
    team.record("CodeRunner-Careful", 1, "code_execute", 0.95)
    team.record("CodeRunner-Literal", 2, "code_execute", 0.8)
    team.record("CodeRunner-Hasty", 3, "code_execute", 0.2)
    assert team.select("code_execute")["id"] in ("CodeRunner-Careful", "CodeRunner-Literal")


def test_retire_then_refine_creates_linked_v2(db):
    bus = Bus(db)
    team = Team(db, bus)
    team.record("CodeRunner-Careful", 1, "code_execute", 0.9)
    team.record("CodeRunner-Literal", 2, "code_execute", 0.8)
    team.record("CodeRunner-Hasty", 3, "code_execute", 0.25, "insufficient: truncated")
    new = team.review("CodeRunner-Hasty")
    assert team.get("CodeRunner-Hasty")["status"] == "retired"
    assert "0.25" in team.get("CodeRunner-Hasty")["retirement_reason"]
    assert new["parent_id"] == "CodeRunner-Hasty" and new["policy"]["epoch_fraction"] == 1.0 and new["avg_reward"] == 0.5


def test_rating_dominates_reward(db):
    team = Team(db, Bus(db))
    team.record("Analyst", 9, "repro_compare", 0.9)
    team.rate(9, 0.0)
    assert team.get("Analyst")["avg_reward"] == pytest.approx(0.315, abs=1e-3)


def test_merge_unions_capabilities_and_keeps_both_parents(db):
    team = Team(db, Bus(db))
    m = team.merge("PaperReader", "Planner")
    assert set(m["capabilities"]) >= {"paper_read", "whatif_design"} and m["parent_id"] == "PaperReader" and m["parent2_id"] == "Planner"
    assert team.get("PaperReader")["status"] == "retired"


def test_fix_memory_only_accepts_data_patches(db):
    mem = Memory(db)
    mem.save_fix("RuntimeError: x", "use cpu", {"patch": {"device": "cpu"}}, 1)
    assert mem.find_fix("RuntimeError: x")["fix"]["patch"] == {"device": "cpu"}
    with pytest.raises(ValueError):
        mem.save_fix("bad", "evil", {"patch": {"__import__": "os"}}, 2)


def test_lessons_are_retrieved_by_relevance_and_cold_start_is_safe(db):
    mem = Memory(db)
    assert mem.related_lessons("anything") == []
    mem.add_lesson("weight decay 0 never grokked within budget", "whatif")
    mem.add_lesson("kaggle kernels need internet enabled", "env")
    assert "weight decay" in mem.related_lessons("what if weight decay is zero")[0]["text"]


def test_executor_parsing_ignores_noise():
    assert parse_kv("METRIC epoch=100 train_acc=0.5 test_acc=0.1")["epoch"] == 100
    assert error_signature(["Traceback...", "  File x", "RuntimeError: CUDA requested"]) == "RuntimeError: CUDA requested"


def test_comparator_names_specific_causes():
    ms = [{"epoch": e, "train_acc": 1.0, "test_acc": a} for e, a in [(100, .3), (500, .3), (900, .97)]]
    eff = {"p": 59, "train_frac": 0.5, "weight_decay": 1.0, "n_layers": 1, "d_model": 128, "batch_size": 0, "lr": 1e-3, "device": "cpu", "seed": 42}
    c = compare("nanda", eff, ms, "2.14.1")
    causes = {x["cause"] for x in c["likely_causes"]}
    assert {"p differs", "hardware / numerics", "random seed"} <= causes and c["qualitative_match"] == "YES"
    assert curve_facts(ms)["grok_epoch"] == 900


def test_planner_requires_hyperparam_search_for_what_ifs():
    p = analyze("reproduce grokking then what if weight decay changes")
    assert "hyperparam_search" in p["capabilities"] and "weight_decay" in p["axes"]
    base = {"train_frac": 0.3, "epochs": 100}
    assert len(design_variants(["weight_decay"], base, sweep=True)) > len(design_variants(["weight_decay"], base, sweep=False))


def test_end_to_end_with_real_subprocess_and_memory():
    """Real train.py subprocess (tiny), real fix recall, real dedup."""
    os.environ["LAB_DB"] = tempfile.mktemp(suffix=".db")
    from lab.controller import Lab

    async def go():
        lab = Lab(DB(os.environ["LAB_DB"]))
        opt = {"preset": "fast", "epochs": 100, "runner": "CodeRunner-Literal"}
        r1 = lab.submit("Reproduce grokking, what if weight decay changes", opt)
        while lab.db.one("SELECT status FROM requests WHERE id=?", (r1,))["status"] not in ("done", "error"):
            await asyncio.sleep(0.5)
        assert lab.db.one("SELECT status FROM requests WHERE id=?", (r1,))["status"] == "done"
        assert lab.db.one("SELECT COUNT(*) n FROM fix_memory")["n"] == 1
        r2 = lab.submit("Reproduce grokking, what if weight decay changes", opt)
        while lab.db.one("SELECT status FROM requests WHERE id=?", (r2,))["status"] not in ("done", "error"):
            await asyncio.sleep(0.5)
        reused = lab.db.one("SELECT COUNT(*) n FROM runs WHERE request_id=? AND reused_from IS NOT NULL", (r2,))["n"]
        assert reused >= 1
        ev = [e["type"] for e in lab.bus.since(0)]
        assert "fix_recalled" in ev and "memory_hit" in ev

    asyncio.run(go())


def test_llm_gateway_meter_cap_and_cache(tmp_path, monkeypatch):
    import lab.llm as L
    from lab.db import DB
    monkeypatch.setenv("ELEVATE_TEAM_API_KEY", "test-key")
    monkeypatch.delenv("LAB_REPLAY", raising=False)
    llm = L.LLM(DB(str(tmp_path / "t.db")))
    calls = []

    def fake(body):
        calls.append(body)
        return {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 123}}
    monkeypatch.setattr(llm, "_post", fake)
    assert llm.ask("sys", "hi") == "hello" and llm.used() == 123
    assert calls[0]["model"] == "gpt-5-mini" and calls[0]["stream"] is False and "max_completion_tokens" in calls[0]
    assert llm.ask("sys", "hi") == "hello" and len(calls) == 1          # second call served from cache, no tokens spent
    monkeypatch.setattr(L, "CAP", 200)
    assert llm.ask("sys", "different") is None and len(calls) == 1      # cap reached: no network call
    assert llm.used() == 123


def test_agent_graph_is_built_from_real_rows(db):
    from lab.taskgraph import agent_graph
    team = Team(db, Bus(db))
    db.x("INSERT INTO runs(id,request_id,label,status,agent_id,attempts,fix_used,result) VALUES(1,1,'baseline','success','CodeRunner-Literal',2,?,?)",
         (json.dumps({"signature": "RuntimeError: CUDA requested", "patch": {"device": "cpu"}, "from_memory": False}), json.dumps({"grok_epoch": 1000})))
    db.x("INSERT INTO runs(id,request_id,label,status,agent_id,attempts) VALUES(2,1,'wd=0','running','CodeRunner-Literal',1)")
    db.x("INSERT INTO task_rewards(agent_id,run_id,capability,base_reward,reward,note,ts) VALUES('CodeRunner-Literal',1,'code_execute',0.9,0.9,'ok',1)")
    g = agent_graph(db, team, "CodeRunner-Literal")
    ids = {n["id"]: n for n in g["nodes"]}
    assert {"agent", "m:code_execute", "t1", "t2"} <= set(ids)
    assert ids["t2"]["status"] == "running" and ids["s2r"]["label"] == "training…"       # live work shows before it is rewarded
    assert ids["s1e"]["status"] == "failed" and "device" in ids["s1f"]["label"]          # error -> fix chain
    assert ids["s1r"]["sub"] == "grokked @ 1000"
    assert ["s1e", "s1f"] in g["edges"] and ["m:code_execute", "t1"] in g["edges"]
    assert agent_graph(db, team, "nobody") is None


def test_restart_recovery_closes_ghosts_and_all_rejected_is_an_error(tmp_path):
    from lab.controller import Lab
    d = DB(str(tmp_path / "r.db"))
    d.x("INSERT INTO requests(id,text,status,created_at) VALUES(1,'x','running',1)")
    d.x("INSERT INTO runs(id,request_id,label,status,agent_id) VALUES(1,1,'baseline','running','CodeRunner-Literal')")
    d.x("INSERT INTO approvals(id,request_id,kind,payload,created_at) VALUES('a1',1,'spawn','{}',1)")

    async def go():
        lab = Lab(d)
        assert d.one("SELECT status FROM runs WHERE id=1")["status"] == "failed"
        assert d.one("SELECT status FROM requests WHERE id=1")["status"] == "error"
        assert d.one("SELECT status FROM approvals WHERE id='a1'")["status"] == "rejected"
        rid = lab.submit("Reproduce grokking", {"preset": "fast", "seed": -5})  # bypasses the API check on purpose
        for _ in range(200):
            await asyncio.sleep(0.1)
            if d.one("SELECT status FROM requests WHERE id=?", (rid,))["status"] in ("error", "done"):
                break
        r = d.one("SELECT status,report FROM requests WHERE id=?", (rid,))
        assert r["status"] == "error" and "rejected" in r["report"]
    asyncio.run(go())
