# Linebreak heads in open-weight LMs

Language models read tokens, yet they put newlines in the right place in fixed-width text. [Gurnee et al. (2025)](https://arxiv.org/abs/2601.04480) showed how Claude 3.5 Haiku does it: it keeps a running character count, and a few attention heads compare that count with the previous line's length. I checked whether open models do the same, across 16 models from Gemma-2, Qwen2.5, Pythia and GPT-2 (160M to 32B).

## What I found

Most do. 15 of the 16 have "twist heads": heads whose query-key scores fall on a diagonal when you plot characters so far against the previous line's length. The exception is Pythia-160M, which is also the worst linebreaker (AUC 0.90).

The heads read the text itself. Models with rotary embeddings could just count tokens since the last newline, but across 57 twist heads in 15 RoPE models the median share of the score coming from RoPE is 0.5%. GPT-2 XL has no RoPE and builds the same heads.

How the work is split depends on model size. Up to about 2.6B parameters the RoPE models have a single twist head, and mean-ablating it makes the model break lines up to 5 characters late (Qwen2.5-0.5B L4H8: 4.8, Gemma-2-2B L13H4: 3.9, Pythia-1.4B L10H3: 3.1). Bigger models have between 3 and 13 of them, tuned to different distances from the end of the line, and removing even the most important one costs only 0.2 to 1.2 characters. GPT-2 XL is the odd one out, with three heads at 1.6B that are each strongly needed. Ablating random heads moves the break point by less than 0.5 characters in every model except one (0.9).

![twist heads vs scale](figures/twist_scaling.svg)

The count is causal. I found a 6-dimensional subspace that tracks the character count and clamped it to read "16 characters earlier". Every model where clamping works at all then breaks 2 to 8 characters later, while clamping a random 6-d subspace moves the break point by at most 0.28. In Gemma-2-2B, editing the query of L13H4 alone gets about half of that effect (2.7 characters; a random edit of the same size gives 0.2).

![steering](figures/steering.svg)

![Gemma-2-2B QK decomposition](figures/gemma2-2b_qk_content_vs_rope.png)
*Gemma-2-2B. Top: pre-RoPE scores, character count vs line length; L13H4 has the diagonal. Bottom: real scores split into content and RoPE parts.*

## What didn't work

Steering only works one way: pushing the count down delays breaks, pushing it up does little (my untested guess is that a count past the line width never appears in training text). Editing L13H4's key, which carries the line length, delays breaks in both directions, so it isn't a width knob. Gemma's count is low-dimensional but smoother than Claude's rippled helix. In GPT-2 XL even a no-op clamp hurts linebreaking, so it's left out of the steering plot.

## Caveats

One run per model, so no error bars. The twist-head thresholds are my own, and I loosened the "effective" rule after Qwen2.5-0.5B's most important head failed it (this changes counts for four Qwen models). Random-head controls didn't exclude weaker twist heads. Qwen's tokenizer merges some newlines with nearby characters, so the Qwen models were scored on 136 samples instead of 400.

## Method

[Pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) prose is re-wrapped greedily at a random width of 20 to 100 characters, so the correct break is always known. The break point is the space left at which P(newline) hits 0.5; effects are shifts in it on 150 held-out samples. A twist head is one where the offset between line length and character count explains over half the variance of its average pre-RoPE scores. It counts as effective if it also attends to the newline and tracks the characters remaining. See `linebreak/mechanism.py`.

## Running it

```bash
uv sync
uv run python run.py --model google/gemma-2-2b                     # ~10 min on an 8 GB GPU
uv run python run.py --model Qwen/Qwen2.5-32B --layer-stride 4      # needs ~80 GB
uv run python run.py --model google/gemma-2-2b --steps count,qk,heads,qk_steer --qk-head 13 4
uv run python scripts/summarize.py && uv run python scripts/plot.py
```

Results for all 16 models are in `results/raw/`. Gemma-2-9B and 27B came from the ungated `unsloth` mirrors. Rerunning reproduces the reported numbers except for the random-control draws.
