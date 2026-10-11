"""Meta-controller brain: which capabilities does a request need, and which experiments follow from it.
Rule-based by default (offline, deterministic). If an LLM is available it may refine the plan, but its output is
re-validated against the fixed capability list and ExperimentConfig, so a bad LLM answer cannot break a run."""
import json, re
from .agents import CAPABILITIES

PRESETS = {  # live-demo-sized overrides; the paper preset keeps the paper's architecture/data split
    "paper": {},
    "fast": {"p": 59, "train_frac": 0.5, "epochs": 3000},
}


def analyze(text: str, llm=None) -> dict:
    t = text.lower()
    caps = ["paper_read", "experiment_plan", "code_execute", "repro_compare", "report_write"]
    whatif = bool(re.search(r"what if|what happens|vary|sweep|different|compare|without|instead|no weight|wd", t)) \
        or "grok" in t or "reproduce" in t
    if whatif:
        caps += ["whatif_design", "hyperparam_search"]
    paper = "power" if re.search(r"\bpower\b", t) and not re.search(r"nanda", t) else "nanda"
    axes = []
    if re.search(r"weight.?decay|\bwd\b|regulari|grok|what if", t):
        axes.append("weight_decay")
    if re.search(r"train.?frac|data|dataset|split|grok", t):
        axes.append("train_frac")
    if re.search(r"\bpower\b|compare|two papers|both", t):
        axes.append("power_setup")
    why = [f"request mentions reproduction/paper -> {', '.join(c for c in caps if c in ('paper_read','experiment_plan'))}"]
    if whatif:
        why.append("open-ended 'what if' investigation -> whatif_design + hyperparam_search")
    plan = {"capabilities": [c for c in CAPABILITIES if c in caps] + ["error_debug"], "paper": paper, "axes": axes or ["weight_decay"], "why": why}
    if llm and llm.available:
        out = llm.ask("Return JSON only: {\"axes\": subset of [\"weight_decay\",\"train_frac\",\"power_setup\"]}. "
                      "Pick the 'what if' axes that best answer the researcher's request.", text)
        try:
            axes_llm = json.loads(re.search(r"\{.*\}", out or "", re.S).group(0)).get("axes", [])
            axes_llm = [a for a in axes_llm if a in ("weight_decay", "train_frac", "power_setup")]
            if axes_llm:
                plan["axes"] = axes_llm
                plan["why"].append("axes refined by LLM")
        except Exception:
            pass
    return plan


def design_variants(axes: list, base: dict, sweep: bool) -> list[dict]:
    """whatif_design chooses axes; hyperparam_search (HyperTuner) expands them into a sweep. Without it, one value per axis."""
    from .config import PAPERS
    out = []
    if "weight_decay" in axes:
        vals = [0.0, 0.1] if sweep else [0.0]
        for v in vals:
            out.append({"label": f"wd={v:g}", "patch": {"weight_decay": v}, "question": f"What if weight decay is {v:g}?"})
    if "train_frac" in axes:
        f = 0.5 if base["train_frac"] < 0.5 else 0.3
        out.append({"label": f"train_frac={f:g}", "patch": {"train_frac": f}, "question": f"What if the model sees {int(f*100)}% of the pairs?"})
    if "power_setup" in axes:
        ps = PAPERS["power"]["setup"]
        patch = {k: ps[k] for k in ("paper", "p", "train_frac", "n_layers", "batch_size")}
        patch["epochs"] = base["epochs"]
        out.append({"label": "power-setup", "patch": patch, "question": "What if we use Power et al.'s setup (2 layers, minibatch 512, p=97)?"})
    return out
