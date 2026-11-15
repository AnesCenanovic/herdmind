# Pilot LoRA fine-tune (backlog item 20)

Status: **two measurements on Colab free (Tesla T4); a T4 with `dtype: float16`
is fast enough. Platform not formally picked; Kaggle has not been measured.**

## What the pilot does

`scripts/pilot_lora.py` runs `distill/pilot.py` with `configs/train/pilot-lora.yaml`:

1. Runs the micro world with the rules brain and reads its `decisions.jsonl`
   (about 4,800 decisions; 2,000 are sampled). No API key is needed.
2. Loads `Mapika/decider-2b`, freezes it and adds LoRA adapters (rank 64, every
   linear layer except the output head), with gradient checkpointing.
3. Trains 60 optimizer steps at batch 8 against the logged distributions
   (KL loss; `labels: hard` switches to cross-entropy on the argmax).
4. Writes `runs/pilot-lora-seed0-<time>/pilot_report.json`: seconds per step,
   tokens per second, seconds per 1,000 rows, peak GPU memory, and what the
   machine supports (GPU, compute capability, bfloat16, Triton, library warnings).

It answers the two open checks left by item 13 (`decider-2b.md`): whether
bfloat16 and the fla/Triton kernels work on a T4-class GPU. A fallback shows up
under `warnings` in the report; an out-of-memory run ends with `status: "oom"`
instead of a crash.

Seconds per step cover forward, backward and the optimizer update only; batch
preparation is left out. Downloading weights from the Hugging Face Hub is the
one network call, made by the `decider`/`transformers` loaders.

The loss values say nothing about model quality: rules-brain labels, 60 steps.

## How to run it (Colab or Kaggle notebook, GPU runtime)

```
!git clone https://github.com/AnesCenanovic/herdmind.git
%cd herdmind
!pip install -q -e ".[train]"
!pip uninstall -q -y torchao
!python scripts/pilot_lora.py configs/train/pilot-lora.yaml
```

Then copy `pilot_report.json` out of the run folder. If the run ends with
`oom`, lower `batch_size` (and raise `accum` to keep the same rows per step);
if bfloat16 is unsupported or very slow, try `dtype: float16`.

The `torchao` line is needed on Colab (2026-11-15 image): it ships
`torchao 0.10.0`, and `peft 0.21` refuses to add LoRA layers while a version
below 0.16 is installed. Nothing here uses `torchao`. The install line worked
on Colab as written; `TODO(verify)` on Kaggle.

The whole run took about 27 minutes on the T4, of which 6 were the first step.
The script logs every step, so do not pipe its output through `tail` if you
want to see progress.

## Results

| Platform | GPU | dtype | batch | status | s/step (median) | s per 1k rows | peak GB | fast kernels |
|---|---|---|---|---|---|---|---|---|
| Colab free | Tesla T4 (15.6 GB) | bfloat16 | 8 | ok | 21.7 | 2,706 | 5.8 | partly (see below) |
| Colab free | Tesla T4 (15.6 GB) | float16 | 8 | ok | 4.89 | 611 | 5.8 | partly (see below) |
| Kaggle | | | | | | | | |

One run per row, seed 0, 2026-11-15, 55 timed steps (5 warm-up steps left out).
The float16 row is the shipped config with only `dtype` changed.
Libraries: torch 2.14.1+cu130, peft 0.21.2, transformers 5.22.1, decider-ai
1.10.0, flash-linear-attention 0.6.0, triton 3.8.0. Weights: commit `7614a6c`
of `Mapika/decider-2b` (null in the first report, read from the download log;
the second report records it).

What the Colab reports say:

- **Memory is not the limit.** 4.1 GB after loading, 5.8 GB peak allocated
  (6.1 GB reserved) of 15.6 GB, with gradient checkpointing on. 67.3 M
  trainable parameters.
  Identical for both dtypes.
- **dtype decides the speed.** bfloat16: 21.7 s per step of 8 rows, about 166
  prompt tokens per second. float16: 4.89 s per step, about 733 tokens per
  second, so 4.4 times faster. Both are very steady (p95 within 1 % of the
  median). Prompts were 405 tokens on average, 412 at most.
- **bfloat16 runs on the T4 but should not be used there.** PyTorch reports
  `bf16_supported: true` (compute capability 7.5), yet it is the slow path.
  This answers the open check from `decider-2b.md`: on a T4, train in float16.
- **The first step took 368 s (bfloat16) and 321 s (float16).** It is left out
  of the timing. Probably one-off kernel compilation; `TODO(verify)`. It is a
  fixed cost per session, about 5 to 6 minutes.
- **Losses are not comparable evidence.** The float16 run ended at 0.17 and
  the bfloat16 run at 0.10, with a noisier first ten steps in float16. One
  seed each and rules-brain labels; item 22 should check float16 training
  quality over several seeds before relying on it.
- **Fast kernels: partly.** Triton imports and `flash-linear-attention` is
  installed, but `transformers` warned that `causal_conv1d_fn` fell back to its
  reference PyTorch implementation ("correct but much slower") because
  `causal_conv1d` is not installed. `flash-attn` is not installed either.

Not measured: `causal_conv1d` installed, gradient checkpointing off (memory
has room for it), a larger batch, and Kaggle. Each could only make the T4
faster than the float16 row.

## Decision

Open. Rule of thumb for reading the table: the full fine-tune (item 26) costs
about `rows x epochs x (s per 1k rows) / 1000` seconds, and must fit inside one
session of the platform with room to spare.

One epoch over 10,000 rows on a free Colab T4, plus about 6 minutes of
start-up: about 7.5 hours in bfloat16 (too slow), about 1.7 hours in float16.

Recommendation: a T4 with `dtype: float16` is enough for item 26 at that
scale, so a free platform will do. Colab free is measured and works; Kaggle
offers the same GPU class with a fixed weekly quota and is unmeasured. Long
runs must save checkpoints, because a free Colab session can end without
warning (the first run's session expired shortly after it finished).

## Provisional choices to confirm before item 22

- **Question text** is the constant `question_text` in the pilot config
  ("What does this animal do next?"), i.e. option A of the open contract
  question in `decider-2b.md`. Not a decision; it only has to exist for the pilot.
- **Briefing** is placed before the perceptions in the context
  (`distill/dataset.py: make_context`).
- **LoRA alpha (128) and target modules** are ours; the card gives only rank
  and learning rate.
