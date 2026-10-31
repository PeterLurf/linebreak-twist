"""results/raw/*.json -> results/summary.csv (one row per model) and results/steering.csv.

    python scripts/summarize.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "results" / "raw", ROOT / "results"

# family, positional scheme, parameters (billions, from model cards)
MODELS = {
    "google/gemma-2-2b": ("Gemma-2", "RoPE", 2.61), "unsloth/gemma-2-9b": ("Gemma-2", "RoPE", 9.24),
    "unsloth/gemma-2-27b": ("Gemma-2", "RoPE", 27.2),
    "Qwen/Qwen2.5-0.5B": ("Qwen2.5", "RoPE", 0.49), "Qwen/Qwen2.5-1.5B": ("Qwen2.5", "RoPE", 1.54),
    "Qwen/Qwen2.5-3B": ("Qwen2.5", "RoPE", 3.09), "Qwen/Qwen2.5-7B": ("Qwen2.5", "RoPE", 7.62),
    "Qwen/Qwen2.5-14B": ("Qwen2.5", "RoPE", 14.7), "Qwen/Qwen2.5-32B": ("Qwen2.5", "RoPE", 32.5),
    "EleutherAI/pythia-160m": ("Pythia", "partial RoPE", 0.16), "EleutherAI/pythia-410m": ("Pythia", "partial RoPE", 0.41),
    "EleutherAI/pythia-1.4b": ("Pythia", "partial RoPE", 1.41), "EleutherAI/pythia-2.8b": ("Pythia", "partial RoPE", 2.78),
    "EleutherAI/pythia-6.9b": ("Pythia", "partial RoPE", 6.86), "EleutherAI/pythia-12b": ("Pythia", "partial RoPE", 11.85),
    "openai-community/gpt2-xl": ("GPT-2", "absolute", 1.56),
}


def effective(h):
    """Twist head that drives real attention: attends to the newline and its score or attention tracks
    characters remaining."""
    return h["twist"] and h["mean_att"] >= 0.01 and (h["eta2_full_rem"] >= 0.3 or h["eta2_att_rem"] >= 0.1)


def main():
    rows, steer = [], []
    for f in sorted(RAW.glob("*.json")):
        r = json.load(open(f))
        fam, pos, params = MODELS.get(r["model"], ("other", "?", float("nan")))
        abl, st = r["twist"]["ablation"], r["steer"]
        base = abl["baseline"]["threshold"]
        single = [k for k in abl if k.startswith("L")]
        rand = [abs(abl[k]["threshold"] - base) for k in abl if k.startswith("random")]
        eff = [h for h in r["twist"]["twist_heads"] if effective(h)]
        d0 = st["d0_clamp_subspace"]["threshold"]
        row = {
            "model": r["model"], "family": fam, "positional": pos, "params_B": params,
            "linebreak_auc": round(r["behavior"]["auc_p_newline"], 4),
            "twist_heads": r["twist"]["n_twist_heads"], "effective_twist_heads": len(eff),
            "effective_heads": " ".join(f"L{h['layer']}H{h['head']}({h['peak_offset']:+.0f})" for h in eff),
            "max_rope_share": round(max((h["rope_var_frac"] for h in eff), default=float("nan")), 3) if pos != "absolute" else "",
            "ablate_best_single": round(min((abl[k]["threshold"] - base for k in single), default=float("nan")), 2),
            "ablate_top4": round(abl["all_twist"]["threshold"] - base, 2) if "all_twist" in abl else "",
            "ablate_random_max_abs": round(max(rand), 2) if rand else "",
            "steer_layers": " ".join(map(str, st["layers"])),
            "clamp_auc_drop": round(st["baseline"]["auc"] - st["d0_clamp_subspace"]["auc"], 3),
        }
        for d in (-16, -8, 8, 16):
            row[f"steer_d{d}"] = round(st[f"d{d}_clamp_subspace"]["threshold"] - d0, 2)
            steer.append({"model": r["model"], "family": fam, "delta": d, "shift": row[f"steer_d{d}"],
                          "random_shift": round(st[f"d{d}_clamp_random_subspace"]["threshold"] - st["baseline"]["threshold"], 2),
                          "clamp_auc_drop": row["clamp_auc_drop"]})
        rows.append(row)
    rows.sort(key=lambda x: (x["family"], x["params_B"]))
    for name, data in (("summary.csv", rows), ("steering.csv", steer)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for r in rows:
        print(f"{r['model']:28s} AUC {r['linebreak_auc']:.3f}  twist {r['effective_twist_heads']:>2}  "
              f"best single ablation {r['ablate_best_single']:+.2f}  steer d-16 {r['steer_d-16']:+.2f}")


if __name__ == "__main__":
    main()
