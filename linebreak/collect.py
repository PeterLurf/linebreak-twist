"""Run the model over wrapped samples and collect what every analysis needs."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

LABELS = ["col", "width", "prev_len", "max_prev_len", "line_idx", "next_word_len", "nl_next", "word_end"]


def newline_id(tok) -> int:
    ids = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, ids
    return ids[0]


@torch.no_grad()
def collect(model, tok, samples, layers, max_positions_per_sample: int, seed: int = 0):
    """Returns dict with
    behavior: per word-end position P(newline) + labels
    resid:    [N, len(layers), d] bfloat16 at sampled in-line positions
    att_last / att_prev: [N, L, H] attention from position to most recent / second most recent newline token
    lab:      dict of label arrays aligned with resid rows (+ 'sample', 'pos', 'tokdist')
    """
    rng = np.random.default_rng(seed)
    nl = newline_id(tok)
    dev = next(model.parameters()).device
    beh = {k: [] for k in LABELS + ["p_nl", "sample"]}
    resid, att_last, att_prev = [], [], []
    lab = {k: [] for k in LABELS + ["sample", "pos", "tokdist", "tok"]}
    for si, s in enumerate(tqdm(samples, desc="collect", mininterval=10)):
        ids = torch.tensor(s.input_ids, device=dev)[None]
        out = model(ids, output_hidden_states=True, output_attentions=True)
        T = ids.shape[1]
        p_nl = torch.softmax(out.logits[0].float(), -1)[:, nl].cpu().numpy()
        # behaviour: word-end decisions after the model has seen >= 2 full lines
        m = s.word_end & (s.line_idx >= 2) & ~s.is_nl
        m[T - 1] = False
        idx = np.nonzero(m)[0]
        beh["p_nl"].append(p_nl[idx]); beh["sample"].append(np.full(len(idx), si))
        for k in LABELS:
            beh[k].append(getattr(s, k)[idx])
        # representation positions: in-line tokens with a previous line available
        cand = np.nonzero((s.line_idx >= 1) & ~s.is_nl & (s.col > 0) & (s.last_nl_tok >= 0))[0]
        if len(cand) > max_positions_per_sample:
            cand = np.sort(rng.choice(cand, max_positions_per_sample, replace=False))
        hs = torch.stack([out.hidden_states[l][0, cand] for l in layers], 1)  # [n, L, d]
        resid.append(hs.to(torch.bfloat16).cpu())  # bf16: Gemma-2-27B residual norms overflow fp16
        att = torch.stack(out.attentions, 0)[:, 0]  # [L, H, T, T]
        ci = torch.tensor(cand, device=dev)
        ln = torch.tensor(s.last_nl_tok[cand], device=dev)
        pn = torch.tensor(np.maximum(s.prev_nl_tok[cand], 0), device=dev)
        has_prev = torch.tensor(s.prev_nl_tok[cand] >= 0, device=dev)
        a_last = att[:, :, ci, ln].permute(2, 0, 1)  # [n, L, H]
        a_prev = att[:, :, ci, pn].permute(2, 0, 1) * has_prev[:, None, None]
        att_last.append(a_last.float().cpu().numpy().astype(np.float16))
        att_prev.append(a_prev.float().cpu().numpy().astype(np.float16))
        for k in LABELS:
            lab[k].append(getattr(s, k)[cand])
        lab["sample"].append(np.full(len(cand), si)); lab["pos"].append(cand)
        lab["tokdist"].append(cand - s.last_nl_tok[cand])
        lab["tok"].append(s.input_ids[cand])
        del out, att
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return dict(behavior=cat(beh), resid=torch.cat(resid), att_last=np.concatenate(att_last),
                att_prev=np.concatenate(att_prev), lab=cat(lab))
