"""Pilot LoRA fine-tune of the student: what does training cost on a given machine?

The pilot answers a platform question, not a quality question. It trains a few
optimizer steps on a small set of logged decisions and writes down seconds per
step, peak GPU memory and whether the GPU supports bfloat16, so the same config
can be run on each candidate platform and the reports compared.

``torch``, ``peft`` and ``decider`` are imported inside functions: they come
with the ``train`` extra, and the rest of the project must import without them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from herdmind.common.config import load_config
from herdmind.common.run import start_run
from herdmind.common.seeding import make_rng
from herdmind.distill.dataset import DistillRow, load_rows
from herdmind.world.experiment import run_experiment
from herdmind.world.spec import WorldSpec

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

REPORT_FILE = "pilot_report.json"
LABEL_KINDS = ("soft", "hard")
DTYPES = ("bfloat16", "float16", "float32")
ALL_LINEAR = "all-linear"
BYTES_PER_GB = 1e9
ROWS_PER_THOUSAND = 1000.0


@dataclass(frozen=True)
class LoraParams:
    """LoRA shape. Rank follows the Decider v12 card.

    TODO(verify): the card gives rank and learning rate only; alpha and the
    target modules are our choice, the vendor's LoRA script is not public.
    """

    rank: int = 64
    alpha: int = 128
    # ("all-linear",) adapts every linear layer except the output head.
    target_modules: tuple[str, ...] = (ALL_LINEAR,)


@dataclass(frozen=True)
class PilotConfig:
    """One pilot run; ``configs/train/pilot-lora.yaml`` holds the values we use."""

    name: str = "pilot-lora"
    seed: int = 0
    model: str = "Mapika/decider-2b"
    # Hub commit of the weights; a run must name the exact files it trained on.
    revision: str = "7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3"
    # Where the rows come from: an existing run folder, or run ``world`` first.
    data_run: str = ""
    world: str = "configs/micro.yaml"
    # Neutral wording of the choice, never advice. TODO(verify): provisional,
    # see "Open contract question" in docs/research/decider-2b.md.
    question_text: str = "What does this animal do next?"
    max_rows: int = 2000
    max_ctx_tokens: int = 1536
    batch_size: int = 8
    steps: int = 60  # optimizer steps
    warmup_steps: int = 5  # first steps left out of timing (kernel compiles, allocator)
    lr: float = 1e-4
    labels: str = "soft"  # soft: KL to the teacher distribution; hard: CE on its argmax
    dtype: str = "float16"
    lora: LoraParams = field(default_factory=LoraParams)

    def __post_init__(self) -> None:
        if self.labels not in LABEL_KINDS:
            raise ValueError(f"labels must be one of {LABEL_KINDS}, got {self.labels!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if min(self.max_rows, self.batch_size) < 1:
            raise ValueError("max_rows and batch_size must be at least 1")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("need 0 <= warmup_steps < steps, or no step is left to time")


@dataclass(frozen=True)
class StepTimings:
    """What the training loop measured.

    ``status`` is "ok", or "oom" when the GPU ran out of memory; an OOM is a
    result of the pilot (this config does not fit this machine), not a crash.
    """

    status: str
    step_seconds: tuple[float, ...]
    step_rows: tuple[int, ...]
    losses: tuple[float, ...]


def distill_loss(
    logits: torch.Tensor, targets: torch.Tensor, nopts: torch.Tensor, labels: str
) -> torch.Tensor:
    """Training loss for one batch of decisions.

    Ground truth is the teacher: its full distribution for ``soft`` (mean
    KL(teacher || student), natural log, the same quantity as
    ``eval.metrics.kl_divergence`` but differentiable and without eps
    clipping), its argmax for ``hard`` (cross-entropy).

    Args:
        logits: ``[N, K]`` student logits; options past ``nopts`` are ``-inf``.
        targets: ``[N, K]`` teacher probabilities, zero past ``nopts``.
        nopts: ``[N]`` number of real options per decision.
        labels: "soft" or "hard".

    Returns:
        Scalar loss.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812 (torch's own convention)

    if labels == "hard":
        return F.cross_entropy(logits, targets.argmax(dim=1))
    valid = torch.arange(logits.shape[1], device=logits.device)[None, :] < nopts[:, None]
    # Padded options have log-prob -inf and target 0; zero both sides so that
    # 0 * -inf does not turn the loss into NaN.
    log_q = torch.log_softmax(logits, dim=1).masked_fill(~valid, 0.0)
    log_p = torch.log(targets.clamp_min(torch.finfo(targets.dtype).tiny))
    return (targets * (log_p - log_q)).masked_fill(~valid, 0.0).sum(dim=1).mean()


def add_lora(lm: torch.nn.Module, params: LoraParams) -> int:
    """Freeze ``lm`` and add trainable LoRA adapters to it, in place.

    Adapters are kept in float32 whatever the base dtype is: in half precision
    a 1e-4 update is partly rounded away, and float16 gradients cannot be
    unscaled. The forward pass mixes the two under autocast.

    Returns:
        Number of trainable parameters.

    Raises:
        RuntimeError: If nothing became trainable, or the output head did: the
            student reads its answer from frozen head rows, so the head must not move.
    """
    import torch
    from peft import LoraConfig, inject_adapter_in_model

    targets = ALL_LINEAR if params.target_modules == (ALL_LINEAR,) else list(params.target_modules)
    cfg = LoraConfig(r=params.rank, lora_alpha=params.alpha, target_modules=targets)
    inject_adapter_in_model(cfg, lm)
    trainable = {n: p for n, p in lm.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError(f"no module matched target_modules={params.target_modules}")
    stray = [n for n in trainable if "lora_" not in n or "lm_head" in n]
    if stray:
        raise RuntimeError(f"unexpected trainable parameters, e.g. {stray[:3]}")
    for p in trainable.values():
        p.data = p.data.to(torch.float32)
    return sum(p.numel() for p in trainable.values())


def train_steps(
    forward: Callable[[dict[str, Any]], torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    batches: Iterator[dict[str, Any]],
    cfg: PilotConfig,
    device: torch.device,
) -> StepTimings:
    """Run ``cfg.steps`` optimizer steps, one batch each, and time each one."""
    import torch

    on_gpu = device.type == "cuda"
    use_amp = cfg.dtype != "float32"  # float32 adapters over a half base need autocast
    # float16 gradients underflow without scaling; bfloat16 has float32's range.
    scaler = torch.amp.GradScaler(device.type, enabled=on_gpu and cfg.dtype == "float16")
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    seconds: list[float] = []
    rows: list[int] = []
    losses: list[float] = []
    status = "ok"
    try:
        for step in range(cfg.steps):
            batch = next(batches)  # before the clock: data loading is not GPU cost
            _sync(device)
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=getattr(torch, cfg.dtype), enabled=use_amp):
                logits = forward(batch)
            if logits.shape != batch["targets"].shape:
                raise RuntimeError(
                    f"model returned logits {tuple(logits.shape)}, "
                    f"expected {tuple(batch['targets'].shape)}"
                )
            loss = distill_loss(logits.float(), batch["targets"], batch["nopts"], cfg.labels)
            scaler.scale(loss).backward()
            if step == 0 and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
                # Seen with re-entrant checkpointing over a frozen base.
                raise RuntimeError("no gradient reached the adapters on the first step")
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            _sync(device)
            seconds.append(time.perf_counter() - start)
            rows.append(int(batch["nopts"].numel()))
            losses.append(float(loss.detach()))
            logger.info(
                "step %d/%d: %.2f s, loss %.4f", step + 1, cfg.steps, seconds[-1], losses[-1]
            )
    except torch.cuda.OutOfMemoryError:
        logger.warning("out of GPU memory at optimizer step %d", len(seconds))
        status = "oom"
    return StepTimings(status, tuple(seconds), tuple(rows), tuple(losses))


def summarize(timings: StepTimings, warmup_steps: int) -> dict[str, Any]:
    """Turn raw step timings into the numbers a platform is judged by.

    The first ``warmup_steps`` are left out of the timing figures because they
    include one-off costs. Pure function, so it is tested without torch.
    """
    out: dict[str, Any] = {
        "status": timings.status,
        "steps_done": len(timings.step_seconds),
        "timed_steps": 0,
        "losses": list(timings.losses),
    }
    seconds = np.asarray(timings.step_seconds[warmup_steps:], dtype=float)
    if seconds.size == 0:
        return out
    rows = sum(timings.step_rows[warmup_steps:])
    out.update(
        timed_steps=int(seconds.size),
        step_seconds_median=float(np.median(seconds)),
        # The figure to scale up from: one pass over 1000 logged decisions.
        seconds_per_1k_rows=ROWS_PER_THOUSAND * float(seconds.sum()) / rows,
    )
    return out


def run_pilot(cfg: PilotConfig, runs_dir: str | Path = "runs") -> Path:
    """Run the pilot and write ``pilot_report.json`` into a new run folder."""
    import torch

    rng = make_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    data_run = cfg.data_run or str(run_experiment(load_config(cfg.world, WorldSpec), runs_dir))
    rows = load_rows(data_run, cfg.max_rows, rng)
    logger.info("pilot rows: %d from %s", len(rows), data_run)

    # Created before the model is touched, so a run that dies while loading
    # still leaves its config and seed behind.
    student = {"brain": "decider", "model_version": f"{cfg.model}@{cfg.revision}"}
    run_dir = start_run(cfg, cfg.name, cfg.seed, {"student": student}, runs_dir)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model, items = _load_student(cfg, rows)
    trainable = add_lora(model.lm, cfg.lora)
    model.to(device).train()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    batches = _batches(items, rows, cfg.batch_size, model.tok.pad_token_id, device, rng)
    params = [p for p in model.parameters() if p.requires_grad]
    timings = train_steps(model, params, batches, cfg, device)

    env: dict[str, Any] = {"cuda": cuda}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        env.update(
            gpu=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            # True on a T4 even though bfloat16 is the slow path there.
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            peak_allocated_gb=torch.cuda.max_memory_allocated() / BYTES_PER_GB,
        )
    report = {
        "environment": env,
        "data": {
            "run": data_run,
            "rows": len(rows),
            "prompt_tokens_max": max(len(it["ids"]) for it in items),
        },
        "model": {**student, "trainable_params": trainable},
        "training": summarize(timings, cfg.warmup_steps),
    }
    (run_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("pilot %s: %s", report["training"]["status"], run_dir / REPORT_FILE)
    return run_dir


@dataclass(frozen=True)
class _PromptQuestion:
    """The shape ``decider.prompt.build`` reads; ``gold`` indexes ``options``."""

    text: str
    options: tuple[str, ...]
    gold: int


@dataclass(frozen=True)
class _PromptExample:
    context: str
    qs: tuple[_PromptQuestion, ...]


class _KeepOrder:
    """Stands in for the RNG ``decider.prompt.build`` shuffles options with.

    Options were already shuffled by ``encoding/`` when the teacher was asked;
    keeping that order keeps teacher probabilities aligned with the prompt.
    """

    def shuffle(self, x: list) -> None:
        return None

    def sample(self, xs: list, k: int) -> list:
        return xs[:k]


def build_items(rows: Sequence[DistillRow], tok: Any, cfg: PilotConfig, chat: Any = None) -> list:
    """Tokenize rows into Decider prompts, one question per row, options in logged order.

    Raises:
        ValueError: If a prompt does not show exactly the row's options in the
            logged order; the teacher probabilities would then label the wrong options.
    """
    from decider.prompt import MAX_OPTIONS, build

    too_wide = [len(r.options) for r in rows if len(r.options) > MAX_OPTIONS]
    if too_wide:
        raise ValueError(f"a row has {too_wide[0]} options, the model reads at most {MAX_OPTIONS}")
    items = [
        build(
            _PromptExample(
                r.context, (_PromptQuestion(cfg.question_text, r.options, r.teacher_argmax),)
            ),
            tok,
            _KeepOrder(),
            max_options=MAX_OPTIONS,
            max_ctx_tokens=cfg.max_ctx_tokens,
            chat=chat,
        )
        for r in rows
    ]
    for row, item in zip(rows, items, strict=True):
        if item["perms"] != [list(range(len(row.options)))]:
            raise ValueError(f"prompt options do not match the logged ones: {item['perms']}")
    return items


def _load_student(cfg: PilotConfig, rows: Sequence[DistillRow]) -> tuple[Any, list]:
    """Fetch the pinned weights, load Decider and tokenize the rows."""
    import torch
    from decider.model import DecisionModel
    from decider.prompt import chat_for_model
    from huggingface_hub import snapshot_download

    # DecisionModel takes no revision, so it loads the pinned snapshot's folder.
    path = snapshot_download(cfg.model, revision=cfg.revision)
    model = DecisionModel(path, dtype=getattr(torch, cfg.dtype), grad_ckpt=False)
    # Non-re-entrant: with a frozen base the re-entrant kind passes no gradient to the adapters.
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    items = build_items(rows, model.tok, cfg, chat=chat_for_model(cfg.model, model.tok))
    return model, items


def _batches(
    items: Sequence[dict],
    rows: Sequence[DistillRow],
    batch_size: int,
    pad_id: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Iterator[dict[str, Any]]:
    """Endless batches in a fresh seeded order per pass, moved to ``device``."""
    import torch
    from decider.model import collate
    from decider.prompt import MAX_OPTIONS

    while True:
        order = rng.permutation(len(items))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = collate([items[j] for j in idx], pad_id)
            targets = torch.zeros((len(idx), MAX_OPTIONS))
            for b, j in enumerate(idx):
                targets[b, : len(rows[j].probs)] = torch.tensor(rows[j].probs)
            batch["targets"] = targets
            yield {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work, so wall time measures the step and not the queue."""
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()
