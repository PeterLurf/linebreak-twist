"""Run the linebreak analyses on one model and write results/raw/<model>.json.

    python run.py --model google/gemma-2-2b
    python run.py --model Qwen/Qwen2.5-7B --layer-stride 2          # store every 2nd layer (memory)
    python run.py --model google/gemma-2-2b --steps count,qk,qk_steer --qk-head 13 4   # Gemma deep dive
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from linebreak import analyses, experiments
from linebreak.collect import collect
from linebreak.data import build_dataset

DEFAULT_STEPS = "behavior,probes,heads,twist,steer"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steps", default=DEFAULT_STEPS, help=f"comma list from {DEFAULT_STEPS},count,qk,qk_steer")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--positions-per-sample", type=int, default=80)
    p.add_argument("--layer-stride", type=int, default=1, help="store residuals every N layers (probes/steering)")
    p.add_argument("--qk-head", type=int, nargs=2, metavar=("LAYER", "HEAD"), help="head for the qk_steer step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    steps = set(a.steps.split(","))
    cfg = {"model": a.model, "widths": [20, 100], "max_tokens": a.max_tokens, "seed": a.seed}
    name = re.sub(r"[^A-Za-z0-9.]+", "-", a.model.split("/")[-1])
    figdir = Path(a.out) / "figures" / name
    figdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager",
                                                 device_map="cuda").eval()
    n = model.config.num_hidden_layers
    layers = list(range(0, n + 1, a.layer_stride))
    samples = build_dataset(tok, a.n_samples, tuple(cfg["widths"]), a.max_tokens, a.seed)
    data = collect(model, tok, samples, layers, a.positions_per_sample, a.seed)
    print(f"{a.model}: {len(samples)} samples, {tuple(data['resid'].shape)} residual rows ({time.time()-t0:.0f}s)", flush=True)

    res = {"model": a.model, "args": vars(a)}
    if "behavior" in steps:
        res["behavior"] = analyses.behavior(data["behavior"], figdir)
        print(f"linebreak AUC {res['behavior']['auc_p_newline']:.3f}", flush=True)
    if "probes" in steps:
        res["probes"] = analyses.probes(data["resid"], data["lab"], layers, figdir)
    if "heads" in steps or "qk" in steps:
        res["heads"] = analyses.heads(data["att_last"], data["lab"], figdir)
    if "count" in steps:
        res["count"] = experiments.run_count(data, layers, figdir)
    if "qk" in steps:
        res["qk"] = experiments.run_qk_decomposition(model, tok, cfg, res["heads"]["top_heads"], figdir)
    if "twist" in steps:
        res["twist"] = experiments.run_twist(model, tok, cfg, figdir)
    if "steer" in steps:
        res["steer"] = experiments.run_steer(model, tok, cfg, data, layers, figdir)
    if "qk_steer" in steps:
        res["qk_steer"] = experiments.run_qk_steer(model, tok, cfg, *a.qk_head)

    raw = Path(a.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(raw / f"{name}.json", "w"), indent=1, default=float)
    print(f"done in {time.time()-t0:.0f}s -> {raw / f'{name}.json'}")


if __name__ == "__main__":
    main()
