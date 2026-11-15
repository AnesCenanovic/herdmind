"""Pilot tests. Nothing here downloads a model: the student is a tiny fake.

Tests that need the ``train`` extra (torch, peft, decider) skip without it, so
the suite still runs on a plain ``uv sync``. They are marked slow because
importing torch and peft alone takes longer than the fast suite allows.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from herdmind.common.config import load_config
from herdmind.distill import pilot
from herdmind.distill.dataset import DistillRow
from herdmind.distill.pilot import LoraParams, PilotConfig, StepTimings, summarize
from herdmind.eval.metrics import kl_divergence

REPO = Path(__file__).resolve().parents[2]
VOCAB = 256
HIDDEN = 16


def test_shipped_config_loads_and_matches_the_card_recipe():
    cfg = load_config(REPO / "configs" / "train" / "pilot-lora.yaml", PilotConfig)
    assert (cfg.lora.rank, cfg.lr, cfg.labels) == (64, 1e-4, "soft")
    assert cfg.lora.target_modules == ("all-linear",)
    assert cfg.revision == PilotConfig().revision  # weights are pinned


@pytest.mark.parametrize(
    "bad", [{"labels": "argmax"}, {"dtype": "int8"}, {"steps": 3, "warmup_steps": 3}]
)
def test_bad_config_fails_loudly(bad):
    with pytest.raises(ValueError):
        PilotConfig(**bad)


def test_summarize_against_a_hand_computed_example():
    # Step 0 is warm-up. Timed: 1 + 2 + 3 = 6 s over 24 rows -> 250 s per 1000 rows.
    timings = StepTimings(
        status="ok",
        step_seconds=(9.0, 1.0, 2.0, 3.0),
        step_rows=(8, 8, 8, 8),
        losses=(1.5, 1.0, 0.8, 0.5),
    )
    s = summarize(timings, warmup_steps=1)
    assert (s["steps_done"], s["timed_steps"]) == (4, 3)
    assert s["step_seconds_median"] == 2.0
    assert s["seconds_per_1k_rows"] == 250.0
    assert s["losses"] == [1.5, 1.0, 0.8, 0.5]


def test_summarize_of_an_early_oom_has_no_timing_figures():
    s = summarize(StepTimings("oom", (), (), ()), warmup_steps=5)
    assert s["status"] == "oom" and s["timed_steps"] == 0 and "step_seconds_median" not in s


# --- needs torch ---------------------------------------------------------------


def _padded(rows, width):
    """Teacher probs and uniform-ish student logits padded to ``width`` options."""
    torch = pytest.importorskip("torch")
    targets = torch.zeros((len(rows), width))
    logits = torch.full((len(rows), width), float("-inf"))
    for i, (p, z) in enumerate(rows):
        targets[i, : len(p)] = torch.tensor(p)
        logits[i, : len(z)] = torch.tensor(z)
    return logits, targets, torch.tensor([len(p) for p, _ in rows])


@pytest.mark.slow
def test_soft_loss_is_kl_teacher_student_and_ignores_padding():
    torch = pytest.importorskip("torch")
    # Row 1 by hand: p = (0.5, 0.5), q = softmax(log 3, 0) = (0.75, 0.25)
    #   KL = 0.5 ln(0.5/0.75) + 0.5 ln(0.5/0.25) = 0.5 ln(4/3).
    # Row 2: student equals teacher, KL = 0. Mean = 0.25 ln(4/3).
    p2 = (0.2, 0.3, 0.5)
    rows = [((0.5, 0.5), (math.log(3.0), 0.0)), (p2, tuple(math.log(x) for x in p2))]
    logits, targets, nopts = _padded(rows, width=6)
    loss = pilot.distill_loss(logits, targets, nopts, "soft")
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.25 * math.log(4 / 3), abs=1e-6)
    # Same number as the project's metric on the unpadded first row.
    assert pilot.distill_loss(logits[:1], targets[:1], nopts[:1], "soft").item() == pytest.approx(
        kl_divergence([[0.5, 0.5]], [[0.75, 0.25]]), abs=1e-6
    )


@pytest.mark.slow
def test_soft_loss_handles_a_teacher_zero():
    torch = pytest.importorskip("torch")
    logits, targets, nopts = _padded([((1.0, 0.0), (0.0, 0.0))], width=4)
    loss = pilot.distill_loss(logits, targets, nopts, "soft")
    assert torch.isfinite(loss) and loss.item() == pytest.approx(math.log(2.0), abs=1e-6)


@pytest.mark.slow
def test_hard_loss_is_cross_entropy_on_the_teacher_argmax():
    pytest.importorskip("torch")
    # Teacher argmax is option 1; student gives it 0.25, so the loss is -ln 0.25.
    logits, targets, nopts = _padded([((0.4, 0.6), (math.log(3.0), 0.0))], width=4)
    loss = pilot.distill_loss(logits, targets, nopts, "hard")
    assert loss.item() == pytest.approx(-math.log(0.25), abs=1e-6)


class _FakeTok:
    """One token per character, so prompts can be decoded back and inspected."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % VOCAB for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _fake_student():
    """A few-thousand-parameter stand-in with Decider's forward signature."""
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class FakeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, HIDDEN)
            self.proj = nn.Linear(HIDDEN, HIDDEN)
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    class FakeStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = FakeLM()
            self.tok = _FakeTok()

        def forward(self, batch):
            h = self.lm.proj(self.lm.emb(batch["input_ids"]))
            hs = h[batch["slot_batch"], batch["slot_idx"]]
            width = batch["targets"].shape[1]
            logits = hs @ self.lm.lm_head.weight[:width].T
            pad = torch.arange(width)[None, :] >= batch["nopts"][:, None]
            return logits.masked_fill(pad, float("-inf"))

    return FakeStudent()


def _rows(n=12):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        k = 3 + i % 3
        p = rng.dirichlet(np.ones(k))
        opts = tuple(f"move {j}" for j in range(k))
        out.append(DistillRow(f"wolf {i} cells north", opts, tuple(p / p.sum())))
    return out


@pytest.mark.slow
def test_add_lora_trains_adapters_only_and_leaves_the_head_frozen():
    pytest.importorskip("peft")
    student = _fake_student()
    n = pilot.add_lora(student.lm, LoraParams(rank=4, alpha=8, target_modules=("proj",)))
    assert n == 2 * 4 * HIDDEN  # A: rank x in, B: out x rank
    assert not student.lm.lm_head.weight.requires_grad
    assert not student.lm.emb.weight.requires_grad


@pytest.mark.slow
def test_adapters_stay_float32_over_a_half_precision_base():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    lm = _fake_student().lm.to(torch.bfloat16)
    pilot.add_lora(lm, LoraParams(rank=4, alpha=8, target_modules=("proj",)))
    dtypes = {p.dtype for p in lm.parameters() if p.requires_grad}
    assert dtypes == {torch.float32} and lm.proj.weight.dtype == torch.bfloat16


@pytest.mark.slow
def test_an_adapter_on_the_output_head_is_refused():
    pytest.importorskip("peft")
    with pytest.raises(RuntimeError, match="unexpected trainable"):
        pilot.add_lora(_fake_student().lm, LoraParams(target_modules=("proj", "lm_head")))


@pytest.mark.slow
def test_add_lora_with_no_matching_module_is_an_error():
    pytest.importorskip("peft")
    with pytest.raises((RuntimeError, ValueError)):
        pilot.add_lora(_fake_student().lm, LoraParams(target_modules=("no_such_layer",)))


@pytest.mark.slow
def test_build_items_keeps_the_logged_option_order():
    pytest.importorskip("decider")
    rows = [DistillRow("ctx", ("stay", "move north", "eat"), (0.2, 0.7, 0.1))]
    tok = _FakeTok()
    (item,) = pilot.build_items(rows, tok, PilotConfig())
    text = tok.decode(item["ids"])
    assert "Question: What does this animal do next?" in text
    assert "(A) stay\n(B) move north\n(C) eat\nAnswer: (" in text
    assert item["golds"] == [1] and item["nopts"] == [3] and item["perms"] == [[0, 1, 2]]


@pytest.mark.slow
def test_build_items_refuses_more_options_than_the_model_reads():
    prompt = pytest.importorskip("decider.prompt")
    k = prompt.MAX_OPTIONS + 1
    row = DistillRow("ctx", tuple(f"o{i}" for i in range(k)), (1.0,) + (0.0,) * (k - 1))
    with pytest.raises(ValueError, match="at most"):
        pilot.build_items([row], _FakeTok(), PilotConfig())


def _run_fake_pilot(tmp_path, monkeypatch, **over):
    pytest.importorskip("peft")
    pytest.importorskip("decider")
    rows = _rows()
    cfg = PilotConfig(
        data_run="unused",
        steps=6,
        warmup_steps=1,
        batch_size=8,
        lr=1e-2,
        dtype="float32",
        lora=LoraParams(rank=4, alpha=8, target_modules=("proj",)),
        **over,
    )
    monkeypatch.setattr(pilot, "load_rows", lambda run, limit, rng: rows)
    monkeypatch.setattr(
        pilot, "_load_student", lambda c, r: (_fake_student(), pilot.build_items(r, _FakeTok(), c))
    )
    run_dir = pilot.run_pilot(cfg, tmp_path)
    return json.loads((run_dir / pilot.REPORT_FILE).read_text())


@pytest.mark.slow
def test_pilot_end_to_end_with_a_fake_student_writes_a_full_report(tmp_path, monkeypatch):
    report = _run_fake_pilot(tmp_path, monkeypatch)
    training = report["training"]
    assert training["status"] == "ok" and training["steps_done"] == 6
    assert training["timed_steps"] == 5 and training["step_seconds_median"] > 0
    assert training["losses"][-1] < training["losses"][0]  # the adapters do learn
    assert report["model"]["model_version"].endswith("@" + PilotConfig().revision)
    assert report["model"]["trainable_params"] == 2 * 4 * HIDDEN
    assert report["data"]["rows"] == 12 and report["data"]["prompt_tokens_max"] > 0
    assert "cuda" in report["environment"]


@pytest.mark.slow
def test_pilot_is_deterministic_under_a_fixed_seed(tmp_path, monkeypatch):
    # The fake student is not seeded by the test: weights and batch order come
    # from the pilot's own seeding.
    a = _run_fake_pilot(tmp_path / "a", monkeypatch, seed=3)
    b = _run_fake_pilot(tmp_path / "b", monkeypatch, seed=3)
    c = _run_fake_pilot(tmp_path / "c", monkeypatch, seed=4)
    assert a["training"]["losses"] == b["training"]["losses"]
    assert a["training"]["losses"] != c["training"]["losses"]
