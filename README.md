# Linebreak heads in open-weight LMs

Language models read tokens, yet they still put newlines in the right place in fixed-width text. [Gurnee et al. (2025)](https://arxiv.org/abs/2601.04480) worked out how Claude 3.5 Haiku does it. The model keeps a running character count, and a few attention heads compare that count with the length of the previous line, attending back to the last newline when the line is nearly full. I wanted to know if open models do the same thing, so I looked at 16 of them: Gemma-2, Qwen2.5, Pythia and GPT-2, from 160M to 32B parameters.

## What I found

Most of them do. In 15 of the 16 models there are heads whose query-key scores line up on a diagonal when you plot "characters so far" against "length of the line that newline ended". Following the original paper I call these twist heads. The exception is Pythia-160M, which is also the worst at linebreaking (AUC 0.90).

The heads work from the text itself. Models with rotary position embeddings could get away with counting tokens since the last newline, and they don't seem to: across 57 twist heads in the 15 RoPE models, the median share of the attention score coming from the RoPE term is 0.5%. GPT-2 XL has no RoPE at all and builds the same kind of head.

How the work is split depends on model size. Up to about 2.6B parameters the RoPE models have a single twist head, and mean-ablating it makes the model break lines up to 5 characters late (Qwen2.5-0.5B L4H8: 4.8, Gemma-2-2B L13H4: 3.9, Pythia-1.4B L10H3: 3.1). Bigger models have between 3 and 13 of them, tuned to different distances from the end of the line, and removing even the most important one costs only 0.2 to 1.2 characters. GPT-2 XL is the odd one out, with three heads at 1.6B that are each strongly needed. Ablating random heads moves the break point by less than 0.5 characters in every model except one (0.9).

![twist heads vs scale](figures/twist_scaling.svg)

The count is causal. I found a 6-dimensional subspace that tracks the character count and clamped it to read "16 characters earlier". Every model where clamping works at all then breaks 2 to 8 characters later, while clamping a random 6-d subspace moves the break point by at most 0.28. In Gemma-2-2B, editing the query of L13H4 alone gets about half of that effect (2.7 characters; a random edit of the same size gives 0.2).

![steering](figures/steering.svg)

![Gemma-2-2B QK decomposition](figures/gemma2-2b_qk_content_vs_rope.png)
*Gemma-2-2B. Top row: pre-RoPE query-key scores with the character count on one axis and line length on the other; L13H4 shows the diagonal. Bottom row: the real score split into its content and RoPE parts. L11H0 tracks token distance through RoPE, but ablating it does nothing.*

## What didn't work

Steering only goes one way. Pushing the count down delays breaks, and pushing it up does very little. My guess is that a count larger than the line width never shows up in wrapped training text, but I haven't tested it. Editing the key side of L13H4, which carries the previous line's length, delays breaks whichever way you push it, so it isn't a clean width knob. Gemma's count representation is low-dimensional but much smoother than the rippled helix reported for Claude. In GPT-2 XL, clamping hurts linebreaking even when the clamp shouldn't change anything, so it's left out of the steering plot.

## Caveats

Each model was run once, so there are no error bars. The twist-head criteria are thresholds I chose. I also loosened the definition of an "effective" head after seeing that Qwen2.5-0.5B's most important head didn't pass the first version; this changes the counts for four Qwen models. The random-head controls didn't exclude weaker twist heads. Qwen's tokenizer merges some newlines with neighbouring characters, so the Qwen models were scored on 136 samples instead of 400.

## Method

Prose from [Pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) is re-wrapped greedily at a random width between 20 and 100 characters, so the right answer is always known: break when `col + 1 + len(next word) > width`. The break point is the amount of space left at which P(newline) reaches 0.5, and every effect is a shift in that point, measured on 150 held-out samples. A head counts as a twist head if, in its matrix of average pre-RoPE scores, the offset between line length and character count explains at least half the variance (η² > 0.5) and the peak moves one-for-one with the count. To count as effective it also has to attend to the newline and have its score or attention track the characters remaining. Details are in `linebreak/mechanism.py`.

## Running it

```bash
uv sync
uv run python run.py --model google/gemma-2-2b                     # ~10 min on an 8 GB GPU
uv run python run.py --model Qwen/Qwen2.5-32B --layer-stride 4      # needs ~80 GB
uv run python run.py --model google/gemma-2-2b --steps count,qk,heads,qk_steer --qk-head 13 4
uv run python scripts/summarize.py && uv run python scripts/plot.py
```

`run.py` writes one JSON file per model to `results/raw/`; `scripts/` turns those into `results/summary.csv` and the figures. The code lives in `linebreak/`. Gemma-2-9B and 27B came from the ungated `unsloth` mirrors. The JSON files here came from an earlier version of this code that also ran some steps I've since dropped, and rerunning reproduces every reported number except the random-control draws.
