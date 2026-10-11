"""Reference grokking trainer (modular addition), ~150 lines, torch only.

Follows Nanda et al. (2023) by default: 1 layer, d=128, 4 heads, d_mlp=512, AdamW(lr=1e-3,
betas=(0.9,0.98), wd=1.0), full-batch, 30% of all (a,b) pairs for training, no LayerNorm.
Power et al. (2022) is the same script with n_layers=2, p=97, batch_size=512.

Output contract (the executor parses ONLY these line types, never free text):
  CONFIG {json}                      the exact config used
  OPTIM  name=AdamW lr=.. wd=..      the exact optimizer kwargs (so weight decay is verifiable)
  METRIC epoch=.. train_acc=.. test_acc=.. train_loss=.. test_loss=..
  RESULT status=grokked|no_grok grok_epoch=.. memorise_epoch=.. final_test_acc=.. sec_per_epoch=.. cpu_s=..
"""
import argparse, json, os, sys, time

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, dm):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, dm), nn.ReLU(), nn.Linear(dm, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(x)


class GrokTransformer(nn.Module):
    def __init__(self, p, d, n_layers, n_heads, d_mlp):
        super().__init__()
        self.embed = nn.Embedding(p + 1, d)  # tokens 0..p-1 plus '=' (index p)
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, d_mlp) for _ in range(n_layers))
        self.head = nn.Linear(d, p, bias=False)
        self.init_like_nanda(p, d)

    @torch.no_grad()
    def init_like_nanda(self, p, d):
        """Nanda et al. draw every weight from N(0, 1/d_model) (unembed: 1/p) with zero biases.
        PyTorch defaults (and N(0,1) embeddings) are much larger and make grokking far slower."""
        for name, prm in self.named_parameters():
            if prm.ndim == 1:
                prm.zero_()
            elif name.startswith("head"):
                prm.normal_(0, 1 / p ** 0.5)
            else:
                prm.normal_(0, 1 / d ** 0.5)

    def forward(self, x):
        t = self.embed(x) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, -1])


def make_data(p, frac, seed, device):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(p * p, generator=g)
    n_tr = int(frac * p * p)
    tr, te = idx[:n_tr], idx[n_tr:]
    return x[tr].to(device), y[tr].to(device), x[te].to(device), y[te].to(device)


@torch.no_grad()
def evaluate(model, x, y):
    out = model(x)
    return F.cross_entropy(out, y).item(), (out.argmax(-1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON experiment config")
    c = json.loads(ap.parse_args().config)
    p, d = c["p"], c["d_model"]
    device = c.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        # what most paper repos hit on a laptop; the debugger agent learns to repair this
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    torch.manual_seed(c["seed"])
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
    print("CONFIG " + json.dumps(c, sort_keys=True), flush=True)

    xtr, ytr, xte, yte = make_data(p, c["train_frac"], c["seed"], device)
    model = GrokTransformer(p, d, c["n_layers"], c["n_heads"], c["d_mlp"]).to(device)
    betas = (0.9, 0.98)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"], betas=betas)
    print(f"OPTIM name=AdamW lr={c['lr']} wd={c['weight_decay']} betas={betas[0]},{betas[1]}", flush=True)

    bs = c.get("batch_size", 0)
    gen = torch.Generator().manual_seed(c["seed"] + 1)
    mem_ep, grok_ep, t0, c0 = None, None, time.time(), time.process_time()
    last = (0.0, 0.0)
    ev = max(1, c.get("eval_every", 50))
    for ep in range(1, c["epochs"] + 1):
        model.train()
        if bs and bs < len(xtr):
            i = torch.randint(0, len(xtr), (bs,), generator=gen).to(device)
            xb, yb = xtr[i], ytr[i]
        else:
            xb, yb = xtr, ytr
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ep % ev == 0 or ep == c["epochs"]:
            model.eval()
            trl, tra = evaluate(model, xtr, ytr)
            tel, tea = evaluate(model, xte, yte)
            last = (tra, tea)
            if mem_ep is None and tra >= 0.99:
                mem_ep = ep
            if grok_ep is None and tea >= 0.95:
                grok_ep = ep
            print(f"METRIC epoch={ep} train_acc={tra:.4f} test_acc={tea:.4f} train_loss={trl:.5f} test_loss={tel:.5f}", flush=True)
            if tea >= 0.99 and tra >= 0.99:
                break
    wall = time.time() - t0
    status = "grokked" if grok_ep else "no_grok"
    print(
        f"RESULT status={status} grok_epoch={grok_ep or 0} memorise_epoch={mem_ep or 0} "
        f"final_test_acc={last[1]:.4f} final_train_acc={last[0]:.4f} epochs_run={ep} "
        f"sec_per_epoch={wall / ep:.5f} cpu_s={time.process_time() - c0:.1f} torch={torch.__version__}",
        flush=True,
    )


if __name__ == "__main__":
    main()
