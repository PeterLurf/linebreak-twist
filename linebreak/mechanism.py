"""Mechanism analyses for fixed-width linebreaking.

- count_controls / count_subspace: the character-count representation (token-identity controlled)
- collect_qk / qk_decomposition:   split a head's score into content (pre-RoPE q.k) and RoPE parts
- twist_scan:                      find "twist" heads in every layer (score organised by width - col)
- eval_breaks + hooks:             causal tests (head ablation, count-subspace clamping, one-head query/key edits)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .analyses import _eta2, _ridge_r2
from .arch import Arch
from .collect import newline_id


# ------------------------------------------------------------------ count representation

def _blocks(model):
    return Arch(model).blocks


def token_centered(resid_l, toks, keep):
    """Subtract the per-token-id mean (over kept rows) from one layer's residuals."""
    X = resid_l[keep].float()
    t = toks[keep]
    uniq, inv = np.unique(t, return_inverse=True)
    inv = torch.tensor(inv)
    sums = torch.zeros(len(uniq), X.shape[1]).index_add_(0, inv, X)
    cnt = torch.bincount(inv, minlength=len(uniq)).float()[:, None]
    return X - (sums / cnt)[inv]


def count_controls(resid, lab, layers, outdir, top_tokens=300, max_col=80, plot_layers=(2, 6, 12)):
    toks = lab["tok"]
    vals, cnts = np.unique(toks, return_counts=True)
    common = set(vals[np.argsort(-cnts)[:top_tokens]].tolist())
    keep = np.array([t in common for t in toks]) & (lab["tokdist"] >= 2)
    col, td, samp = lab["col"][keep], lab["tokdist"][keep], lab["sample"][keep]
    # character count beyond what token count predicts, and vice versa
    e_col_td = {v: col[td == v].mean() for v in np.unique(td)}
    col_beyond = col - np.array([e_col_td[v] for v in td])
    test = torch.tensor(np.isin(samp, np.unique(samp)[::5])); train = ~test
    halfA = np.isin(samp, np.unique(samp)[::2])
    res = {"n_rows": int(keep.sum()), "probe": {}, "cv_pca": {}}
    cols = np.array([c for c in range(2, max_col + 1)
                     if (col[halfA] == c).sum() >= 15 and (col[~halfA] == c).sum() >= 15])
    figs = {}
    for li, l in enumerate(layers):
        X = token_centered(resid[:, li], toks, keep)
        res["probe"][l] = {name: max(_ridge_r2(X, y, train, test, lam) for lam in (1e-2, 1e-1))
                           for name, y in [("col", col), ("tokdist", td), ("col_beyond_tokdist", col_beyond)]}
        # split-half class means
        MA = torch.stack([X[torch.tensor(halfA & (col == c))].mean(0) for c in cols])
        MB = torch.stack([X[torch.tensor(~halfA & (col == c))].mean(0) for c in cols])
        MA, MB = MA - MA.mean(0), MB - MB.mean(0)
        _, S, V = torch.linalg.svd(MA, full_matrices=False)
        totB = (MB ** 2).sum()
        ve = [float(((MB @ V[:k].T) ** 2).sum() / totB) for k in range(1, 31)]
        # reliability ceiling: fraction of B's variance shared with A at all
        shared = float((MA * MB).sum() / torch.sqrt((MA ** 2).sum() * totB))
        res["cv_pca"][l] = {"cv_ve_k": {k: ve[k - 1] for k in (1, 2, 3, 4, 6, 8, 10, 15, 20, 30)},
                            "split_half_corr": shared}
        if l in plot_layers:
            figs[l] = (MA, MB, V)
    # figure: cross-half cosine similarity (noise-free ringing) + top-PC trajectories
    fig, axes = plt.subplots(2, len(figs), figsize=(3.6 * len(figs), 7), squeeze=False)
    for j, (l, (MA, MB, V)) in enumerate(sorted(figs.items())):
        A = MA / MA.norm(dim=1, keepdim=True); B = MB / MB.norm(dim=1, keepdim=True)
        C = (A @ B.T).numpy()
        ax = axes[0, j]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=[cols[0], cols[-1], cols[-1], cols[0]])
        ax.set_title(f"layer {l}: cos(mean_A[i], mean_B[j])", fontsize=8)
        ax.set_xlabel("col j (half B)"); ax.set_ylabel("col i (half A)")
        P = (MB @ V[:4].T).numpy()
        ax = axes[1, j]
        for k in range(4):
            ax.plot(cols, P[:, k], label=f"PC{k+1}")
        ax.set_xlabel("col"); ax.set_title(f"layer {l}: half-B means on half-A PCs", fontsize=8); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(outdir / "count_controls.png", dpi=150); plt.close(fig)
    # probe figure
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name in ("col", "tokdist", "col_beyond_tokdist"):
        ax.plot(layers, [res["probe"][l][name] for l in layers], "o-", ms=3, label=name)
    ax.set_ylim(0, 1); ax.set_xlabel("layer"); ax.set_ylabel("held-out R^2 (token-centred)")
    ax.legend(fontsize=7); ax.set_title("Character count beyond token count")
    fig.tight_layout(); fig.savefig(outdir / "count_probes_controlled.png", dpi=150); plt.close(fig)
    return res


def count_subspace(resid, lab, layers, layer, k, max_col=80):
    """Top-k PCs of token-centred per-col means at `layer`, plus raw (uncentred) per-col means."""
    li = layers.index(layer)
    toks = lab["tok"]
    vals, cnts = np.unique(toks, return_counts=True)
    common = set(vals[np.argsort(-cnts)[:300]].tolist())
    keep = np.array([t in common for t in toks]) & (lab["tokdist"] >= 2)
    X = token_centered(resid[:, li], toks, keep)
    col = lab["col"][keep]
    cols = np.arange(1, max_col + 1)
    M = torch.stack([X[torch.tensor(col == c)].mean(0) if (col == c).sum() >= 10 else torch.full((X.shape[1],), float("nan"))
                     for c in cols])
    ok = ~torch.isnan(M[:, 0])
    Mc = M[ok] - M[ok].mean(0)
    _, _, V = torch.linalg.svd(Mc, full_matrices=False)
    Xr = resid[:, li][keep].float()
    Mraw = torch.stack([Xr[torch.tensor(col == c)].mean(0) if (col == c).sum() >= 10 else torch.full((X.shape[1],), float("nan"))
                        for c in cols])
    return cols, M, V[:k], Mraw


# ------------------------------------------------------------------ QK decomposition

@torch.no_grad()
def collect_qk(model, samples, heads_by_layer: dict[int, list[int]], max_pos=80, seed=0):
    rng = np.random.default_rng(seed)
    A = Arch(model)
    grabs = {}
    hooks = A.capture_attn_inputs(list(heads_by_layer), grabs)
    dev = next(model.parameters()).device
    keys_all = [(l, h) for l, hs in heads_by_layer.items() for h in hs]
    out = {kk: {"content": [], "full": [], "att": []} for kk in keys_all}
    qs = {kk: [] for kk in keys_all}; nlk = {kk: [] for kk in keys_all}
    lab = {k: [] for k in ("col", "prev_len", "tokdist", "sample")}
    nl_len = []
    try:
        for si, s in enumerate(samples):
            res = model(torch.tensor(s.input_ids, device=dev)[None], output_attentions=True)
            cand = np.nonzero((s.line_idx >= 1) & ~s.is_nl & (s.col > 0) & (s.last_nl_tok >= 0))[0]
            if len(cand) > max_pos:
                cand = np.sort(rng.choice(cand, max_pos, replace=False))
            key = s.last_nl_tok[cand]
            nl_pos = np.nonzero(s.is_nl)[0]
            for l, hs in heads_by_layer.items():
                x, pe = grabs[l]
                q, k = A.qk(l, x[0].float())
                qr, kr = A.rope(q, pe), A.rope(k, pe)
                for h in hs:
                    kv = A.kv_of(h)
                    out[(l, h)]["content"].append((q[cand, h] * k[key, kv]).sum(-1).cpu())
                    out[(l, h)]["full"].append((qr[cand, h] * kr[key, kv]).sum(-1).cpu())
                    out[(l, h)]["att"].append(res.attentions[l][0, h, cand, key].float().cpu())
                    qs[(l, h)].append(q[cand, h].cpu())
                    nlk[(l, h)].append(k[nl_pos, kv].cpu())
            lab["col"].append(s.col[cand]); lab["prev_len"].append(s.prev_len[cand])
            lab["tokdist"].append(cand - key); lab["sample"].append(np.full(len(cand), si))
            nl_len.append(s.col[nl_pos - 1])
    finally:
        for hk in hooks:
            hk.remove()
    out = {kk: {n: torch.cat(v).numpy() for n, v in d.items()} for kk, d in out.items()}
    for kk in out:
        out[kk]["q"] = torch.cat(qs[kk]); out[kk]["nl_k"] = torch.cat(nlk[kk])
    return out, {k: np.concatenate(v) for k, v in lab.items()}, np.concatenate(nl_len)


def qk_decomposition(qk, lab, nl_len, outdir, max_col=90):
    """Which part of the pre-softmax score tracks chars remaining: content (pre-RoPE) or RoPE?"""
    rem = lab["prev_len"] - lab["col"]
    groups = {"remaining": np.clip(rem, -10, 60) // 2, "col": np.clip(lab["col"], 0, 100) // 2,
              "tokdist": np.clip(lab["tokdist"], 0, 40), "prev_len": np.clip(lab["prev_len"], 0, 100) // 2}
    res = {}
    keys = sorted(qk)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.4 * len(keys), 6.4), squeeze=False)
    cols = np.arange(5, max_col + 1)
    for j, kk in enumerate(keys):
        d = qk[kk]
        content, full = d["content"], d["full"]
        rope = full - content
        r = {}
        for part, y in (("full", full), ("content", content), ("rope", rope)):
            r[part] = {g: float(_eta2(y[:, None], gv)[0]) for g, gv in groups.items()}
            r[part]["var"] = float(y.var())
        r["corr_full_content"] = float(np.corrcoef(full, content)[0, 1])
        r["corr_full_rope"] = float(np.corrcoef(full, rope)[0, 1])
        # twist matrix from class means of pre-RoPE q (by col) and newline k (by line length)
        q, nk = d["q"], d["nl_k"]
        qm = torch.stack([q[torch.tensor(lab["col"] == c)].mean(0) for c in cols])
        km = torch.stack([nk[torch.tensor(nl_len == w)].mean(0) if (nl_len == w).sum() > 3
                          else torch.full_like(qm[0], float("nan")) for w in cols])
        S = (qm @ km.T).numpy()
        Sc = S - np.nanmean(S, axis=1, keepdims=True)
        ii, ww = np.meshgrid(cols, cols, indexing="ij")
        ok = ~np.isnan(Sc)
        r["twist"] = {"eta2_by_offset(w-i)": float(_eta2(Sc[ok][:, None], (ww - ii)[ok] // 2)[0]),
                      "eta2_by_w": float(_eta2(Sc[ok][:, None], ww[ok] // 2)[0])}
        peak = cols[np.nanargmax(np.where(ok, Sc, -np.inf), axis=1)]
        r["twist"]["peak_w_minus_i_median"] = float(np.median(peak - cols))
        r["twist"]["peak_w_vs_i_slope"] = float(np.polyfit(cols, peak, 1)[0])
        name = f"L{kk[0]}H{kk[1]}"
        res[name] = r
        ax = axes[0, j]
        ax.imshow(Sc, origin="lower", cmap="RdBu_r", extent=[cols[0], cols[-1], cols[0], cols[-1]], aspect="auto")
        ax.plot(cols, cols, "k--", lw=0.6)
        ax.set_xlabel("key: newline ending a line of length w"); ax.set_ylabel("query: col i")
        ax.set_title(f"{name} pre-RoPE q.k (row-centred)", fontsize=8)
        ax = axes[1, j]
        xs = np.arange(-10, 61, 2)
        for part, y in (("full score", full), ("content (pre-RoPE)", content), ("RoPE part", rope)):
            ys = [y[(rem >= x) & (rem < x + 2)].mean() if ((rem >= x) & (rem < x + 2)).sum() > 10 else np.nan for x in xs]
            ax.plot(xs, np.array(ys) - np.nanmean(ys), label=part)
        ax.set_xlabel("prev_len - col"); ax.set_ylabel("score (centred)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(outdir / "qk_decomposition.png", dpi=150); plt.close(fig)
    return res


# ------------------------------------------------------------------ twist scan over every head

@torch.no_grad()
def twist_scan(model, samples, outdir, max_pos=60, max_col=90, min_col=5, seed=0):
    """For every head: class-mean pre-RoPE twist matrix S[i, w] = qbar(col=i) . kbar(newline ending line of len w),
    plus content-vs-RoPE decomposition of the real scores from in-line tokens to the most recent newline."""
    rng = np.random.default_rng(seed)
    A = Arch(model)
    L, H, dh = A.n_layers, A.H, A.dh
    dev = next(model.parameters()).device
    grabs = {}
    hooks = A.capture_attn_inputs(range(L), grabs)
    C = max_col + 1
    qsum = torch.zeros(L, C, H, dh, device=dev); qcnt = torch.zeros(C, device=dev)
    ksum = torch.zeros(L, C, H, dh, device=dev); kcnt = torch.zeros(C, device=dev)
    content, full, att = [], [], []
    labs = {k: [] for k in ("col", "prev_len", "tokdist")}
    kv_idx = torch.tensor([A.kv_of(h) for h in range(H)], device=dev)
    try:
        for s in samples:
            res = model(torch.tensor(s.input_ids, device=dev)[None], output_attentions=True)
            cand = np.nonzero((s.line_idx >= 1) & ~s.is_nl & (s.col > 0) & (s.last_nl_tok >= 0))[0]
            if len(cand) > max_pos:
                cand = np.sort(rng.choice(cand, max_pos, replace=False))
            key = s.last_nl_tok[cand]
            nl_pos = np.nonzero(s.is_nl)[0]
            nl_len = s.col[nl_pos - 1]
            okq = s.col[cand] <= max_col
            okk = nl_len <= max_col
            ci = torch.tensor(s.col[cand][okq], device=dev)
            wi = torch.tensor(nl_len[okk], device=dev)
            qcnt.index_add_(0, ci, torch.ones_like(ci, dtype=torch.float))
            kcnt.index_add_(0, wi, torch.ones_like(wi, dtype=torch.float))
            c_l, f_l, a_l = [], [], []
            for l in range(L):
                x, pe = grabs[l]
                q, k = A.qk(l, x[0].float())
                k = k[:, kv_idx]  # expand GQA to per-query-head keys
                qr, kr = A.rope(q, pe), A.rope(k, pe)
                c_l.append((q[cand] * k[key]).sum(-1))
                f_l.append((qr[cand] * kr[key]).sum(-1))
                a_l.append(res.attentions[l][0][:, cand, key].T.float())
                qsum[l].index_add_(0, ci, q[cand][okq])
                ksum[l].index_add_(0, wi, k[nl_pos][okk])
            content.append(torch.stack(c_l, 1).cpu()); full.append(torch.stack(f_l, 1).cpu()); att.append(torch.stack(a_l, 1).cpu())
            labs["col"].append(s.col[cand]); labs["prev_len"].append(s.prev_len[cand]); labs["tokdist"].append(cand - key)
            del res
    finally:
        for hk in hooks:
            hk.remove()
    content, full, att = (torch.cat(v).numpy() for v in (content, full, att))  # [N, L, H]
    labs = {k: np.concatenate(v) for k, v in labs.items()}
    rem = labs["prev_len"] - labs["col"]
    N = content.shape[0]
    remb = np.clip(rem, -10, 60) // 2
    tdb = np.clip(labs["tokdist"], 0, 40)
    e = lambda y, g: _eta2(y.reshape(N, -1), g).reshape(L, H)
    rope = full - content
    stats = {
        "eta2_full_rem": e(full, remb), "eta2_content_rem": e(content, remb), "eta2_rope_rem": e(rope, remb),
        "eta2_att_rem": e(att, remb), "eta2_att_tokdist": e(att, tdb), "mean_att": att.mean(0),
        "rope_var_frac": rope.var(0) / (full.var(0) + 1e-9),
    }
    cols = np.arange(min_col, max_col + 1)
    okc = (qcnt[cols].cpu().numpy() >= 10) & (kcnt[cols].cpu().numpy() >= 5)
    cols = cols[okc]
    qm = (qsum[:, cols] / qcnt[cols][None, :, None, None]).cpu()  # [L, n, H, dh]
    km = (ksum[:, cols] / kcnt[cols][None, :, None, None]).cpu()
    ii, ww = np.meshgrid(cols, cols, indexing="ij")
    offs = ((ww - ii) // 2).ravel()
    tw = {k: np.zeros((L, H)) for k in ("eta2_offset", "eta2_w", "peak_offset", "peak_slope")}
    mats = {}
    for l in range(L if len(cols) >= 10 else 0):  # need enough populated classes for a twist matrix
        S = torch.einsum("ihd,jhd->hij", qm[l], km[l]).numpy()  # [H, i, w]
        for h in range(H):
            Sc = S[h] - S[h].mean(1, keepdims=True)
            tw["eta2_offset"][l, h] = _eta2(Sc.ravel()[:, None], offs)[0]
            tw["eta2_w"][l, h] = _eta2(Sc.ravel()[:, None], (ww // 2).ravel())[0]
            peak = cols[Sc.argmax(1)]
            tw["peak_offset"][l, h] = np.median(peak - cols)
            tw["peak_slope"][l, h] = np.polyfit(cols, peak, 1)[0]
            mats[(l, h)] = Sc
    # a "twist head": the score is organised by the offset w - i (diagonal band), not by w alone
    twist = (tw["eta2_offset"] > 0.5) & (np.abs(tw["peak_slope"] - 1) < 0.3)
    rows = []
    for l in range(L):
        for h in range(H):
            rows.append({"layer": l, "head": h, "twist": bool(twist[l, h]),
                         **{k: float(v[l, h]) for k, v in tw.items()},
                         **{k: float(v[l, h]) for k, v in stats.items()}})
    rows.sort(key=lambda r: -r["eta2_offset"])
    # figures
    top = [r for r in rows if r["twist"]][:8] or rows[:8]
    if not mats:
        top = []
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, r in zip(axes.ravel(), top):
        Sc = mats[(r["layer"], r["head"])]
        ax.imshow(Sc, origin="lower", cmap="RdBu_r", extent=[cols[0], cols[-1], cols[0], cols[-1]], aspect="auto")
        ax.plot(cols, cols, "k--", lw=0.5)
        ax.set_title(f"L{r['layer']}H{r['head']}  offset {r['peak_offset']:+.0f}  eta2 {r['eta2_offset']:.2f}", fontsize=8)
        ax.set_xlabel("key: line length w", fontsize=7); ax.set_ylabel("query: col i", fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "twist_scan_top.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.arange(-10, 61, 2)
    for r in top:
        a = att[:, r["layer"], r["head"]]
        ys = [a[(rem >= x) & (rem < x + 2)].mean() if ((rem >= x) & (rem < x + 2)).sum() > 10 else np.nan for x in xs]
        ys = np.array(ys)
        ax.plot(xs, ys / np.nanmax(ys), label=f"L{r['layer']}H{r['head']}")
    ax.set_xlabel("prev_len - col"); ax.set_ylabel("attention to last newline (max-normalised)")
    ax.legend(fontsize=6); ax.set_title("Do twist heads tile 'chars remaining'?")
    fig.tight_layout(); fig.savefig(outdir / "twist_tiling.png", dpi=140); plt.close(fig)
    return {"n_twist_heads": int(twist.sum()), "twist_heads": [r for r in rows if r["twist"]], "top_by_offset": rows[:20],
            "all_heads": rows}


# ------------------------------------------------------------------ causal tests

@torch.no_grad()
def eval_breaks(model, tok, samples, per_sample_hooks=None):
    """P(newline) at word-end decisions with line_idx >= 2.
    per_sample_hooks(sample) -> list of hook handles installed for that sample's forward."""
    nl = newline_id(tok)
    dev = next(model.parameters()).device
    P, Y, SL = [], [], []
    for s in samples:
        handles = per_sample_hooks(s) if per_sample_hooks else []
        try:
            logits = model(torch.tensor(s.input_ids, device=dev)[None]).logits[0].float()
        finally:
            for hd in handles:
                hd.remove()
        p = torch.softmax(logits, -1)[:, nl].cpu().numpy()
        m = s.word_end & (s.line_idx >= 2) & ~s.is_nl
        m[len(s.input_ids) - 1] = False
        idx = np.nonzero(m)[0]
        P.append(p[idx]); Y.append(s.nl_next[idx]); SL.append((s.width - s.col - 1 - s.next_word_len)[idx])
    P, Y, SL = map(np.concatenate, (P, Y, SL))
    nll = -np.mean(np.where(Y, np.log(P + 1e-9), np.log(1 - P + 1e-9)))
    thr, slope = threshold(P, SL)
    return dict(auc=float(roc_auc_score(Y, P)), break_nll=float(nll), threshold=thr, slope=slope,
                acc=float(((P > 0.5) == Y).mean()))


def threshold(p, slack):
    """Slack at which mean P(newline) crosses 0.5 (linear interp of the binned curve)."""
    xs = np.arange(-12, 13)
    ys = np.array([p[slack == x].mean() if (slack == x).sum() > 5 else np.nan for x in xs])
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    z = np.log(np.clip(ys, 1e-4, 1 - 1e-4) / (1 - np.clip(ys, 1e-4, 1 - 1e-4)))
    a, b = np.polyfit(xs, z, 1)
    return float(-b / a), float(a)


def head_mean_ablation_hooks(model, heads_by_layer):
    """Replace the chosen heads' outputs (pre-o_proj) with their mean over the sequence."""
    A = Arch(model)
    dh = A.dh

    def make(l, hs):
        def pre(mod, args):
            x = args[0].clone()
            for h in hs:
                sl = slice(h * dh, (h + 1) * dh)
                x[..., sl] = x[..., sl].mean(dim=1, keepdim=True)
            return (x,) + args[1:]
        return pre

    def install(_s):
        return [A.out_proj(l).register_forward_pre_hook(make(l, hs))
                for l, hs in heads_by_layer.items()]
    return install


def clamp_hooks_multi(model, layer_specs):
    """layer_specs = [(layer, V [k,d], target [max_col+1, k])]: at in-line tokens set the subspace coordinates
    of hidden_states[layer] to target[col] (idempotent, so safe to apply at several layers)."""
    dev = next(model.parameters()).device
    blocks = _blocks(model)
    specs = [(blocks[L - 1], V.to(dev), T.to(dev)) for L, V, T in layer_specs]

    def install(s):
        mask = (s.line_idx >= 1) & ~s.is_nl & (s.col > 0) & (s.col < specs[0][2].shape[0])
        colv = torch.tensor(np.where(mask, s.col, 0), device=dev)
        m = torch.tensor(mask, device=dev)
        handles = []
        for blk, V, T in specs:
            def hook(mod, args, out, V=V, T=T):
                hs = out[0] if isinstance(out, tuple) else out
                x = hs[0].float()
                tgt = T[colv]
                valid = m & ~torch.isnan(tgt[:, 0])
                newx = x - (x @ V.T) @ V + torch.nan_to_num(tgt) @ V
                x = torch.where(valid[:, None], newx, x)
                hs = x[None].to(hs.dtype)
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            handles.append(blk.register_forward_hook(hook))
        return handles
    return install


def clamp_targets(Mraw, V, delta, max_col):
    """target[c] = V @ Mraw[c + delta] (subspace coords of the raw class mean for col c+delta)."""
    T = torch.full((max_col + 1, V.shape[0]), float("nan"))
    for c in range(1, max_col + 1):
        c2 = c + delta
        if 1 <= c2 <= max_col and not torch.isnan(Mraw[c2 - 1, 0]):
            T[c] = V @ Mraw[c2 - 1]
    return T


# ------------------------------------------------------------------ one-head query / key edits

def qk_mean_tables(qk_entry, lab, nl_len, max_col, min_n=8):
    """Class means of pre-RoPE query by col and newline key by line length (NaN rows where sparse)."""
    q, nk = qk_entry["q"], qk_entry["nl_k"]
    dh = q.shape[1]
    Q = torch.full((max_col + 1, dh), float("nan")); K = torch.full((max_col + 1, dh), float("nan"))
    for c in range(1, max_col + 1):
        m = lab["col"] == c
        if m.sum() >= min_n:
            Q[c] = q[torch.tensor(m)].mean(0)
        m = nl_len == c
        if m.sum() >= min_n:
            K[c] = nk[torch.tensor(m)].mean(0)
    return Q, K


def delta_table(M, delta):
    T = torch.zeros_like(M)
    for c in range(M.shape[0]):
        c2 = c + delta
        if 0 <= c2 < M.shape[0] and not torch.isnan(M[c, 0]) and not torch.isnan(M[c2, 0]):
            T[c] = M[c2] - M[c]
    return T


def q_shift_hooks(model, layer, head, table):
    """Add table[col] to one head's pre-RoPE query (q_proj output) at in-line tokens of lines >= 1."""
    A = Arch(model)
    dev = next(model.parameters()).device
    tab = table.to(dev)
    sl = slice(head * A.dh, (head + 1) * A.dh)

    def install(s):
        mask = (s.line_idx >= 1) & ~s.is_nl & (s.col > 0) & (s.col < tab.shape[0])
        colv = torch.tensor(np.where(mask, s.col, 0), device=dev)

        def hook(mod, args, out):
            out = out.clone()
            out[0, :, sl] += tab[colv].to(out.dtype)
            return out
        return [A.attn(layer).q_proj.register_forward_hook(hook)]
    return install


def k_shift_hooks(model, layer, kv_head, table):
    """Add table[w] to one kv head's pre-RoPE key at each newline token, w = length of the line it ends."""
    A = Arch(model)
    dev = next(model.parameters()).device
    tab = table.to(dev)
    sl = slice(kv_head * A.dh, (kv_head + 1) * A.dh)

    def install(s):
        w = np.zeros(len(s.input_ids), dtype=np.int64)
        nl = np.nonzero(s.is_nl)[0]
        w[nl] = np.clip(s.col[nl - 1], 0, tab.shape[0] - 1)
        wv = torch.tensor(w, device=dev)

        def hook(mod, args, out):
            out = out.clone()
            out[0, :, sl] += tab[wv].to(out.dtype)
            return out
        return [A.attn(layer).k_proj.register_forward_hook(hook)]
    return install
