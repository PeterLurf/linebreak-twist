"""Behaviour, linear probes and the attention-to-newline head scan."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def behavior(beh, outdir):
    y = beh["nl_next"].astype(int)
    p = beh["p_nl"]
    slack = beh["width"] - beh["col"] - 1 - beh["next_word_len"]  # <0 => must break
    res = {
        "n_decisions": int(len(y)),
        "base_rate_break": float(y.mean()),
        "auc_p_newline": float(roc_auc_score(y, p)),
        "acc_at_0.5": float(((p > 0.5) == y).mean()),
        "mean_p_nl_when_break": float(p[y == 1].mean()),
        "mean_p_nl_when_no_break": float(p[y == 0].mean()),
    }
    bins = np.arange(-15, 16)
    curve = [float(p[slack == b].mean()) if (slack == b).sum() > 5 else None for b in bins]
    res["p_nl_by_slack"] = dict(zip(bins.tolist(), curve))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(bins, [c if c is not None else np.nan for c in curve], "o-", ms=3)
    ax.axvline(-0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("slack = width - col - 1 - len(next word)")
    ax.set_ylabel("mean P(newline)")
    ax.set_title("Linebreak behaviour (break iff slack < 0)")
    fig.tight_layout(); fig.savefig(outdir / "behavior_slack.png", dpi=150); plt.close(fig)
    return res


def _ridge_r2(X, y, train, test, lam):
    X = X.float().cuda(); y = torch.tensor(y, dtype=torch.float32, device="cuda")
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-4
    X = (X - mu) / sd
    ym = y[train].mean()
    Xt, yt = X[train], y[train] - ym
    A = Xt.T @ Xt + lam * len(Xt) * torch.eye(X.shape[1], device="cuda")
    w = torch.linalg.solve(A, Xt.T @ yt)
    pred = X[test] @ w + ym
    r2 = 1 - ((pred - y[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()
    return float(r2)


def probes(resid, lab, layers, outdir):
    samp = lab["sample"]
    test = torch.tensor(np.isin(samp, np.unique(samp)[::5]))
    train = ~test
    targets = {
        "col": lab["col"],
        "prev_len": lab["prev_len"],
        "remaining(prev_len-col)": lab["prev_len"] - lab["col"],
        "tokdist": lab["tokdist"],
    }
    res = {}
    for name, y in targets.items():
        res[name] = {}
        for li, l in enumerate(layers):
            res[name][l] = max(_ridge_r2(resid[:, li], y, train, test, lam) for lam in (1e-3, 1e-2, 1e-1))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name in targets:
        ax.plot(layers, [res[name][l] for l in layers], "o-", ms=3, label=name)
    ax.set_xlabel("layer (hidden_states index)"); ax.set_ylabel("held-out R^2"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7); ax.set_title("Linear probes")
    fig.tight_layout(); fig.savefig(outdir / "probes.png", dpi=150); plt.close(fig)
    return res


def _eta2(x, g):
    """Fraction of variance of each column of x explained by group means of g."""
    tot = ((x - x.mean(0)) ** 2).sum(0)
    ss = np.zeros_like(tot)
    for v in np.unique(g):
        m = g == v
        if m.sum() < 10:
            continue
        ss += m.sum() * (x[m].mean(0) - x.mean(0)) ** 2
    return ss / (tot + 1e-9)


def heads(att_last, lab, outdir, top=12):
    """Boundary-head scan: how much of a head's attention to the previous newline is explained
    by (prev_len - col) beyond what col alone explains."""
    N, L, H = att_last.shape
    a = att_last.astype(np.float32)
    rem = lab["prev_len"] - lab["col"]
    col = lab["col"]
    e_rem = _eta2(a.reshape(N, -1), np.clip(rem, -10, 60) // 2).reshape(L, H)
    e_col = _eta2(a.reshape(N, -1), np.clip(col, 0, 100) // 2).reshape(L, H)
    mean_att = a.mean(0)
    score = np.where(mean_att > 0.02, e_rem - e_col, -1)
    order = np.dstack(np.unravel_index(np.argsort(-score.ravel()), (L, H)))[0][:top]
    rows = [{"layer": int(l), "head": int(h), "mean_att_prev_nl": float(mean_att[l, h]),
             "eta2_remaining": float(e_rem[l, h]), "eta2_col": float(e_col[l, h])} for l, h in order]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = np.arange(-10, 61, 2)
    for r in rows[:6]:
        l, h = r["layer"], r["head"]
        ys = [a[(rem >= x) & (rem < x + 2), l, h].mean() if ((rem >= x) & (rem < x + 2)).sum() > 10 else np.nan
              for x in xs]
        ax.plot(xs, ys, label=f"L{l}H{h}")
    ax.set_xlabel("prev_len - col (chars remaining vs previous line)"); ax.set_ylabel("attention to last newline")
    ax.legend(fontsize=7); ax.set_title("Candidate boundary heads")
    fig.tight_layout(); fig.savefig(outdir / "boundary_heads.png", dpi=150); plt.close(fig)
    return {"top_heads": rows}
