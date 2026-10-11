"""ReproComparator: says WHERE and WHY our result differs from the paper, with evidence.
Deterministic: every cause is derived from a concrete difference (config diff, epoch cap, curve shape, environment)."""
from .config import PAPERS


def curve_facts(metrics):
    mem = next((m["epoch"] for m in metrics if m["train_acc"] >= 0.99), None)
    grok = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.95), None)
    plateau = [m["test_acc"] for m in metrics if mem and m["epoch"] >= mem and (not grok or m["epoch"] < grok)]
    return {"memorise_epoch": mem, "grok_epoch": grok, "plateau_test_acc": round(sum(plateau) / len(plateau), 3) if plateau else None,
            "final_test_acc": metrics[-1]["test_acc"] if metrics else None, "last_epoch": metrics[-1]["epoch"] if metrics else 0}


def compare(paper_key: str, eff: dict, metrics: list, torch_version: str | None = None) -> dict:
    paper = PAPERS[paper_key]
    setup, claim = paper["setup"], paper["claim"]
    f = curve_facts(metrics)
    causes = []
    for k in ("p", "train_frac", "weight_decay", "n_layers", "d_model", "batch_size", "lr"):
        if k in setup and eff.get(k) != setup[k]:
            causes.append({"cause": f"{k} differs", "evidence": f"paper {setup[k]} vs ours {eff.get(k)}", "strength": "config"})
    claimed = claim.get("grok_epoch")
    if claimed and f["grok_epoch"] is None and f["last_epoch"] < claimed:
        causes.append({"cause": "epoch cap", "evidence": f"we stopped at epoch {f['last_epoch']}, before the paper's reported {claimed}", "strength": "config"})
    elif claimed and f["grok_epoch"] is not None and abs(f["grok_epoch"] - claimed) / claimed > 0.25:
        causes.append({"cause": "grok timing differs", "evidence": f"ours at {f['grok_epoch']} vs reported ~{claimed}; paper hyperparameters/seed not fully specified", "strength": "inferred"})
    if eff.get("device") != paper.get("paper_device"):
        causes.append({"cause": "hardware / numerics", "evidence": f"paper ran on {paper['reports']['hardware']}, we ran on {eff.get('device')}", "strength": "environment"})
    causes.append({"cause": "random seed", "evidence": f"paper seed {paper['reports']['seed'] or 'not reported'}, ours {eff.get('seed')}", "strength": "environment"})
    if torch_version:
        causes.append({"cause": "library version", "evidence": f"paper PyTorch version {paper['reports']['torch_version'] or 'not reported'}, ours {torch_version}", "strength": "environment"})
    order = {"config": 0, "inferred": 1, "environment": 2}
    causes.sort(key=lambda c: order[c["strength"]])
    grokked = f["grok_epoch"] is not None
    qual = "YES" if grokked else ("PARTIAL" if f["memorise_epoch"] else "NO")
    return {"paper": paper["title"], "paper_key": paper_key, "claim_verified": claim.get("verified", False), "claim_note": claim.get("note"),
            "paper_reports": {"grok_epoch": claimed, "final_test_acc": claim.get("final_test_acc")},
            "we_observed": f, "qualitative_match": qual, "likely_causes": causes}


def summary_line(cmp: dict) -> str:
    r, o = cmp["paper_reports"], cmp["we_observed"]
    top = "; ".join(f"{c['cause']} ({c['evidence']})" for c in cmp["likely_causes"][:3])
    rep = f"~{r['grok_epoch']}" if r["grok_epoch"] else "n/a"
    got = o["grok_epoch"] if o["grok_epoch"] else f"no grok by {o['last_epoch']}"
    tag = "" if cmp["claim_verified"] else " [claim unverified]"
    return f"Paper reports grok at {rep}{tag}; we observed {got}. Likely: {top}."
