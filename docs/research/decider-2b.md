# Research: Mapika Decider 2B (backlog item 13)

Status: first pass, 2026-11-08. Sources are the Hugging Face model card, the
`Mapika/decider` GitHub repo (read directly: `train.py`, `train.sh`,
`calibrate.py`, `prompt.py`, `model.py`, `infer.py`) and one smoke test on a
laptop. Anything marked `TODO(verify)` is vendor-reported or not yet tested by us.

## Summary

- Decider 2B is a usable student model. Weights are public, Apache 2.0, and it
  returns one probability per option in a single forward pass, which matches the
  `Brain` contract.
- Inference runs on a 6 GB Windows laptop GPU in eager mode, but without the fast
  path, and only at small batch sizes.
- Training will not fit on that laptop. Plan to train on a cloud GPU.
- The repo's own training script is a **full fine-tune with hard labels**, not LoRA
  and not soft labels. Soft-label distillation (our plan for item 26) needs a small
  custom loss.
- The model needs a **question text** that our `Question` contract does not
  carry. See "Open contract question".

## Weights and licence

| Item | Finding |
|---|---|
| Model | `Mapika/decider-2b` (v12), 2B dense, BF16 weights file 3.8 GB |
| Base | Qwen3.5-2B-Base (Gated DeltaNet + attention hybrid, 24 layers) |
| Weights licence | Apache 2.0, same as the base model |
| Code licence | Apache 2.0 (`decider-ai` package, repo `Mapika/decider`) |
| Training data | Mixed terms; some sets are non-commercial (ScienceQA, IconQA, TQA, RACE) |
| Access | Public files, no gating notice seen |

The non-commercial training sources are irrelevant for a university project but
should be mentioned in the final write-up. Aggregator sites disagree with the model
card about base models and package versions; this page uses only the card and the
repo.

## What the model does

It is not a text generator. It reads a state and typed questions, each with an
explicit option list, and returns a probability distribution per question.

- Prompt layout (plain text), rendered by `decider/prompt.py`:

  ```
  Context:
  <state text>

  Question: <question text>
  Options:
  (A) <option 1>
  (B) <option 2>
  Answer: (
  ```

- Readout (`decider/model.py`): the hidden state at the final `(` is projected onto
  the LM-head rows of the letter tokens (A, B, C, ...) and softmaxed over the valid
  options. Letters are never generated.
- Up to 255 options for a choice question; our questions have 3 to 8.
- The default context cap is 1536 tokens of state text.
- Question types: `choice`, `noul` (yes/no), `score`. We only need `choice`.
- Inference (`Decider.decide_batch`) does **not** shuffle options (`_NoShuffle`),
  so shuffling and un-shuffling stay in `encoding/`, as our rules require.
  Options containing words like "other" or "unsure" may be renamed internally
  (`neutralize_none`); avoid those words as action labels.

### Calibration

Probabilities are temperature-scaled per question type. Card values (fitted by the
authors on their own validation rows): choice 2.465, score 2.833, noul 3.185.
The default config sets yes/no to temperature 0.05, which makes answers near 0 or 1
and breaks calibration. We use the fitted choice temperature rather than the
default. `decider/calibrate.py` can refit temperatures on our own data: it takes
`records.jsonl` lines of `{"type", "logits" or "probs", "gold"}` and minimises NLL,
per type, with a fallback to a default if a type has fewer than 50 rows.

## Training

What the card says (v12): LoRA on the language model only, rank 64, learning rate
1e-4, 2 epochs, about 56 minutes on one B300, 39.5k text rows plus 18k image rows.
`TODO(verify)`: the script for that recipe is not in the repo.

What the repo contains (`decider/train.py`, `scripts/train.sh`):

- Loss: `F.cross_entropy` against one integer gold index per question, with optional
  label smoothing and an optional Brier term. **No soft-label or KL path.**
- Full fine-tune: `model.parameters()` goes straight to AdamW (lr 1e-5, betas
  0.9/0.95, cosine schedule). No PEFT, no gradient checkpointing call, no explicit
  dtype in the script.
- Batching by token budget (default 16384 tokens per micro-batch, accumulation 2).
- `train.sh` targets one GH200 or H100-class GPU and about 1.3M examples.
- Training rows come from `decider.prompt.build(...)`:
  `dict(ids, slots, golds, nopts, perms)`, where `perms` maps displayed option
  order back to original order.

### What this means for soft vs hard labels (item 26)

`DecisionModel` returns raw logits `[N, MAX_OPTIONS]` with invalid options masked
to `-inf`, and the loss lives outside the module. A KL loss against the teacher
distribution is therefore a small custom loss. The teacher probabilities must be
reordered into the displayed order using `perms`. The stock cross-entropy recipe
is the hard-label baseline for the comparison.

`DecisionModel` adds no trainable parameters, so LoRA is a matter of wrapping
`lm.model` with PEFT. It also supports gradient checkpointing.

## Smoke test on a Windows laptop (2026-11-08)

Machine: RTX 4050 Laptop GPU with **6 GB VRAM**, 23 GB RAM, Windows 11. A throwaway
Python 3.12 `uv` environment outside the repo. PyTorch 2.14.1+cu130,
`decider-ai` 1.10.0, `transformers` 5.22.2, `flash-linear-attention` 0.6.0.

| Check | Result |
|---|---|
| Model loads in BF16 | Yes, 3.76 GB allocated. First load 128 s (download), cached load 8 s |
| Default fast path (CUDA graphs compiled with Triton) | **Fails** with `TritonMissing`. Triton has no official Windows build |
| Eager mode (`use_graphs=False`) | Works. `flash-linear-attention` cannot use its kernels without Triton, so Transformers falls back to slower reference PyTorch code and warns about it |
| Output is a valid distribution | Yes: one value per option (4 and 5 options), sums to 1.000000 |
| Deterministic | Yes: identical probabilities on repeated calls |
| Option-order sensitivity | Reversing the option order moved individual probabilities by up to about 4 points on one toy prompt (0.251 to 0.211). Small test, but it supports the shuffle rule |
| Latency, batch 8 | 0.24 s (about 30 ms per decision) |
| Latency, batch 32 | 1.1 s (about 34 ms per decision) |
| Latency, batch 64 | 11 s (about 173 ms per decision) |
| Peak VRAM, batch 64 | 6.39 GB allocated on a 6.14 GB card |

The batch-64 slowdown is consistent with Windows spilling into shared system memory.
That is an inference, not something we measured directly. On this machine the batch
size should be capped around 32. The probabilities themselves are from a toy prompt
I wrote and say nothing about model quality; real behaviour is for items 18 and 22.

Unofficial Triton-for-Windows builds might restore the fast path. Not tested.

## Where to train

Weights alone are 3.8 GB, so LoRA training with activations does not fit in 6 GB.
Options, from the most to the least tested:

1. **This laptop, native Windows**: inference only (see above).
2. **WSL2 on this laptop**: WSL2 is installed but the only distro is
   `docker-desktop`. Still a 6 GB GPU, so it would fix Triton, not memory.
   Not tested.
3. **Google Colab (free)**: sources say the free tier usually assigns a T4 with about
   15 to 16 GB. Google publishes no fixed quota and says free resources are not
   guaranteed; session limits quoted by third parties vary (12 h session cap, 90 min
   idle timeout). `TODO(verify)` against Google's FAQ.
4. **Kaggle notebooks (free)**: community-reported weekly GPU quota of about 30 hours,
   with P100 (16 GB) or T4 x2 (2 x 16 GB). The quota is not in Kaggle's docs and the
   session cap is reported as 9 h in some places and 12 h in others.
   `TODO(verify)` against Kaggle's own pages.

Two unresolved risks apply to both free options:

- **bfloat16.** T4 (Turing) lacks native BF16 tensor-core support and P100 is older.
  The model is shipped in BF16, so training or inference there may need FP16 or run
  slowly. `TODO(verify)` against NVIDIA documentation.
- **Triton and `flash-linear-attention` on pre-Ampere GPUs.** One forum post (secondhand,
  not confirmed in the fla repo) says fla kernels target compute capability 8.0 or
  newer. If true, a T4 would fall back to the slow reference implementation, which is
  slower and more memory-hungry for training. We found no direct statement either
  way. `TODO(verify)`: run a short Gated DeltaNet forward and backward pass on a T4 in
  Colab and on a Kaggle GPU before choosing a platform. That test belongs to item 20.

## Open contract question: question text (needs a decision)

Decider's prompt contains a question line (`Question: ...`) next to the options.
Our `Question` dataclass (`encoding/question.py`) carries only `state_text` and
`options`. The wording has to come from somewhere. Options:

- **A. A constant inside the Decider brain.** No contract change. The Jev brain and
  the Decider brain must still use the same wording, or we measure the wording
  instead of the model.
- **B. A new field on `Question`.** Explicit and shared, but it changes a shared
  contract.

This page recommends **A for now** and lists B as the alternative; the choice is not
made here. Per the project rules, any change to `Question` needs agreement from the
people who work on `encoding/` and the brains first. Whatever the wording is, it
must describe the choice neutrally ("What does this animal do next?") and must not
carry advice. The distilled model must be trained on the same wording it is used with.

## Recommendations

1. Item 20 (pilot LoRA): run the T4 check above first. If fla/Triton do not work on
   pre-Ampere GPUs, look at an Ampere-or-newer GPU (Colab paid, Kaggle if it offers
   one, or a university or rented GPU) before spending effort on the pilot.
2. Item 22 (Decider brain): use eager mode on Windows, cap the batch size (around 32 on
   a 6 GB card), pass options in the order `encoding/` produced, and use the fitted
   choice temperature.
3. Item 26: plan two loss variants, stock cross-entropy on the teacher's argmax and
   a custom KL loss on the teacher's distribution. Refit temperatures with
   `calibrate.py` on our own logs.
4. Resolve the open contract question before item 22 starts.

## Open questions

- Whether the v12 LoRA recipe script is available anywhere (not in the repo we read).
- Real VRAM for LoRA training with gradient checkpointing (card gives none).
- Whether the T4 and P100 can run the fla kernels.
- Quality on our task. All benchmark numbers on the card (for example Decision
  Index 38.17) are self-reported by the authors, `TODO(verify)`.

## Sources

- [Mapika/decider-2b model card](https://huggingface.co/Mapika/decider-2b)
- [Mapika on Hugging Face](https://huggingface.co/Mapika)
- [Mapika/decider repo](https://github.com/Mapika/decider)
- [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- Colab and Kaggle limits (third-party, not official): [Hivenet Colab guide](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations), [GPU Per Hour free GPUs](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [GMI Cloud free GPU options](https://www.gmicloud.ai/en/blog/best-free-gpu-cloud-options-for-ai-startups-and-researchers)
- fla on older GPUs (forum, secondhand): [PyTorch forum thread](https://discuss.pytorch.org/t/silent-kernel-death/225653)
