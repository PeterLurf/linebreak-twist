"""Experiment drivers called from run.py. Each returns a JSON-serialisable dict and prints a short summary."""
from __future__ import annotations

import json

import numpy as np
import torch

from . import mechanism as mech
from .analyses import _ridge_r2
from .data import build_dataset


def _eval_set(tok, cfg, n, seed_offset=1):
    return build_dataset(tok, n, tuple(cfg["widths"]), cfg["max_tokens"], cfg["seed"] + seed_offset)


def run_count(data, layers, out):
    """Character count beyond token count (token-identity centred) + split-half PCA of per-col means."""
    return mech.count_controls(data["resid"], data["lab"], layers, out)


def run_qk_decomposition(model, tok, cfg, head_rows, out, n_heads=4):
    """Content (pre-RoPE) vs RoPE split of the score for the heads whose newline attention best tracks
    chars remaining (from the attention head scan)."""
    top = [(r["layer"], r["head"]) for r in head_rows if r["mean_att_prev_nl"] > 0.02][:n_heads]
    hbl = {}
    for l, h in top:
        hbl.setdefault(l, []).append(h)
    qk, lab, nl_len = mech.collect_qk(model, _eval_set(tok, cfg, 100), hbl)
    return mech.qk_decomposition(qk, lab, nl_len, out)


def run_twist(model, tok, cfg, out, n_scan=100, n_eval=150, n_ablate=4, n_random=3):
    """Twist scan over every head, then mean-ablate the top twist heads (singly and together) vs random heads."""
    eval_samples = _eval_set(tok, cfg, n_eval)
    ts = mech.twist_scan(model, eval_samples[:n_scan], out)
    print(f"twist heads: {ts['n_twist_heads']}", flush=True)
    cand = [r for r in ts["twist_heads"] if r["mean_att"] > 0.01][:n_ablate]
    abl = {"baseline": mech.eval_breaks(model, tok, eval_samples)}
    if cand:
        hbl = {}
        for r in cand:
            hbl.setdefault(r["layer"], []).append(r["head"])
            abl[f"L{r['layer']}H{r['head']}"] = mech.eval_breaks(
                model, tok, eval_samples, mech.head_mean_ablation_hooks(model, {r["layer"]: [r["head"]]}))
        abl["all_twist"] = mech.eval_breaks(model, tok, eval_samples, mech.head_mean_ablation_hooks(model, hbl))
        chosen = {(r["layer"], r["head"]) for r in cand}
        lo, hi = min(chosen)[0], max(chosen)[0]
        pool = [(r["layer"], r["head"]) for r in ts["all_heads"] if (r["layer"], r["head"]) not in chosen and lo <= r["layer"] <= hi]
        rng = np.random.default_rng(cfg["seed"])
        for i in range(n_random):
            pick = [pool[j] for j in rng.choice(len(pool), len(cand), replace=False)]
            d = {}
            for l, h in pick:
                d.setdefault(l, []).append(h)
            r = mech.eval_breaks(model, tok, eval_samples, mech.head_mean_ablation_hooks(model, d))
            r["heads"] = [f"L{l}H{h}" for l, h in pick]
            abl[f"random_{i}"] = r
    for k, v in abl.items():
        print(f"  ablate {k:12s} threshold {v['threshold']:+.2f}", flush=True)
    return {"twist_heads": ts["twist_heads"], "n_twist_heads": ts["n_twist_heads"], "ablation": abl}


def run_steer(model, tok, cfg, data, layers, out, k=6, deltas=(-16, -8, 8, 16), n_eval=150, max_col=80):
    """Clamp the k-dim count subspace to the class mean for col + delta at up to 5 layers (those where a linear
    col probe reaches >= 90% of its best R^2); control = a random k-dim subspace clamped the same way."""
    samp = data["lab"]["sample"]
    test = torch.tensor(np.isin(samp, np.unique(samp)[::5]))
    r2 = {l: _ridge_r2(data["resid"][:, li], data["lab"]["col"], ~test, test, 1e-2) for li, l in enumerate(layers) if l > 0}
    good = [l for l in sorted(r2) if r2[l] >= 0.9 * max(r2.values())]
    sl = [good[i] for i in np.unique(np.linspace(0, len(good) - 1, min(5, len(good))).round().astype(int))]
    subs = {L: mech.count_subspace(data["resid"], data["lab"], layers, L, k, max_col) for L in sl}  # cols, M, V, Mraw
    eval_samples = _eval_set(tok, cfg, n_eval)
    g = torch.Generator().manual_seed(cfg["seed"])
    res = {"layers": sl, "baseline": mech.eval_breaks(model, tok, eval_samples)}

    def clamp(delta, V_of):
        specs = [(L, V_of(L), mech.clamp_targets(subs[L][3], V_of(L), delta, max_col)) for L in sl]
        return mech.eval_breaks(model, tok, eval_samples, mech.clamp_hooks_multi(model, specs))

    res["d0_clamp_subspace"] = clamp(0, lambda L: subs[L][2])
    for delta in deltas:
        res[f"d{delta}_clamp_subspace"] = clamp(delta, lambda L: subs[L][2])
        rand = {L: torch.linalg.qr(torch.randn(subs[L][2].shape[1], k, generator=g))[0].T for L in sl}
        res[f"d{delta}_clamp_random_subspace"] = clamp(delta, lambda L: rand[L])
        print(f"  steer delta={delta:+d}: shift {res[f'd{delta}_clamp_subspace']['threshold'] - res['d0_clamp_subspace']['threshold']:+.2f}"
              f" (random subspace {res[f'd{delta}_clamp_random_subspace']['threshold'] - res['baseline']['threshold']:+.2f})", flush=True)
    return res


def run_qk_steer(model, tok, cfg, layer, head, deltas=(-16, -8, 8, 16), max_col=90):
    """Edit only one head: add qbar(col+delta) - qbar(col) to its pre-RoPE query at in-line tokens, or
    kbar(w+delta) - kbar(w) to its key at newline tokens; control = equal-norm random direction."""
    Q, K = mech.qk_mean_tables(*_qk_fit(model, tok, cfg, layer, head), max_col)
    eval_samples = _eval_set(tok, cfg, 150)
    kv = mech.Arch(model).kv_of(head)
    g = torch.Generator().manual_seed(cfg["seed"])
    res = {"baseline": mech.eval_breaks(model, tok, eval_samples)}
    for delta in deltas:
        for side, M, hooks in (("query", Q, lambda T: mech.q_shift_hooks(model, layer, head, T)),
                               ("key", K, lambda T: mech.k_shift_hooks(model, layer, kv, T))):
            T = mech.delta_table(M, delta)
            u = torch.randn(T.shape[1], generator=g); u /= u.norm()
            for name, tab in ((side, T), (side + "_random", T.norm(dim=1, keepdim=True) * u[None])):
                res[f"d{delta}_{name}"] = mech.eval_breaks(model, tok, eval_samples, hooks(tab))
                print(f"  {name:13s} delta={delta:+d}: shift {res[f'd{delta}_{name}']['threshold'] - res['baseline']['threshold']:+.2f}", flush=True)
    return res


def _qk_fit(model, tok, cfg, layer, head):
    qk, lab, nl_len = mech.collect_qk(model, _eval_set(tok, cfg, 150, seed_offset=2), {layer: [head]})
    return qk[(layer, head)], lab, nl_len
