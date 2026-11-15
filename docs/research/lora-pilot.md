# Pilot LoRA fine-tune (backlog item 20)

Status: tooling ready, **no measurements yet, platform not picked**. The pilot
has only been run against a tiny fake model on a CPU (the tests). The real run
needs a GPU; fill in the table below from each platform's `pilot_report.json`.

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
!python scripts/pilot_lora.py configs/train/pilot-lora.yaml
```

Then copy `pilot_report.json` out of the run folder. If the run ends with
`oom`, lower `batch_size` (and raise `accum` to keep the same rows per step);
if bfloat16 is unsupported or very slow, try `dtype: float16`.

`TODO(verify)`: the install line is untested on Colab and Kaggle. Both ship
their own PyTorch; if `pip` replaces it with a build that does not match the
runtime's CUDA, install `decider-ai` and `peft` separately instead of the extra.

## Results

| Platform | GPU | dtype | batch | status | s/step (median) | s per 1k rows | peak GB | fast kernels |
|---|---|---|---|---|---|---|---|---|
| Colab free | | | | | | | | |
| Kaggle | | | | | | | | |

## Decision

Open. Rule of thumb for reading the table: the full fine-tune (item 26) costs
about `rows x epochs x (s per 1k rows) / 1000` seconds, and must fit inside one
session of the platform with room to spare.

## Provisional choices to confirm before item 22

- **Question text** is the constant `question_text` in the pilot config
  ("What does this animal do next?"), i.e. option A of the open contract
  question in `decider-2b.md`. Not a decision; it only has to exist for the pilot.
- **Briefing** is placed before the perceptions in the context
  (`distill/dataset.py: make_context`).
- **LoRA alpha (128) and target modules** are ours; the card gives only rank
  and learning rate.
