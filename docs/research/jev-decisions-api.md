# Research: Jev Decisions API (backlog item 10)

Status: first pass, 2026-11-14. Sources are OpenRouter's API reference for
`POST /api/alpha/decisions`, OpenRouter's Jev tutorial, two OpenRouter blog posts
and the LiteLLM decisions page (all read directly, see "Sources"). The Jev model
page on OpenRouter returned 404 when fetched. **No live call was made**: there was
no API key on the machine used for this write-up, so the raw-response checks below
are still `TODO(verify)`. Every number here is vendor-reported unless stated
otherwise.

## Summary

- **The key question has a good answer.** A `choice` answer carries `choice`,
  `confidence` and `probabilities`, an object mapping **every option name to a
  probability**. Every documented example lists all options. Soft-label distillation
  (item 26) is possible.
- But the schema marks `probabilities` as optional, and every published value has
  **two decimals** (0.84 / 0.16 / 0). If Jev really rounds, the teacher distribution
  has 0.01 resolution and small probabilities become exact zeros. That matters for
  KL and for soft labels. `TODO(verify)` on a raw response.
- **One state per request.** A request can carry many questions, but they all share
  one `state`. In our world each animal has its own perceptions, so it is one request
  per animal, with requests sent in parallel.
- **Cheap.** Input tokens only: $0.042 per million (TODO(verify) on the model
  page). Assuming about 500 input tokens per decision (not measured yet), one
  decision costs about $0.00002 and a 50k-decision experiment about $1.
- **Not deterministic.** Probabilities vary slightly from call to call, and there is
  no seed parameter. Reproducibility has to come from the disk cache (item 9).
- **The documentation is silent on** rate limits, maximum options per question,
  maximum questions per request, option-order sensitivity and streaming.

## Endpoint and models

| Item | Finding |
|---|---|
| Endpoint | `POST https://openrouter.ai/api/alpha/decisions`, bearer key. It is not the chat-completions endpoint, so chat SDKs do not work with it |
| Stability | Labelled **alpha**. The request and response shape may change |
| SDKs | Official `alpha.decisions.create` in the Python, TypeScript and Go SDKs. Our code uses `requests` and does not need them |
| Models | `typesafe/jev-1.13` (text only), `typesafe/jev-1.14` (released 2026-10-14, adds images and `box`), alias `~typesafe/jev-latest` |
| Version echo | The response field `model` gives the dated snapshot, e.g. `typesafe/jev-1.13-20260917` |
| Context | Jev 1.13: 64k tokens per request, of which 32k for state plus the longest question. A long-context mode (262k) switches on above about 32,000 bytes of text, and TypeSafe calls it "slower and less accurate". Our states are a few hundred bytes |

## Request

```json
{
  "model": "typesafe/jev-1.13",
  "state": "string, or a JSON object or array",
  "questions": {
    "<id>": {
      "type": "choice",
      "instructions": "optional question text",
      "criteria": {"<option name>": "guidance (string/object/array/null)"}
    }
  }
}
```

- Question types: `noul` (yes/no), `choice`, `score` (ordered levels, cap 10),
  `box` (1.14 only). We only need `choice`.
- The question id is **not sent to the model**. Meaning has to live in
  `instructions` and in the per-option `criteria` text, which Jev reads literally.
- Optional top-level fields: `provider` (routing preferences, e.g. `max_price`,
  `zdr`), `session_id`, `trace`, `user`, `assets` (images).
- No `temperature`, `seed`, `stream` or cache-control field exists in the schema.

## Response

Example from the API reference (abridged):

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "provider": "TypeSafe",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "payments",
      "confidence": 0.75,
      "probabilities": {"account": 0, "frontend": 0.16, "payments": 0.84}
    }
  },
  "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992}
}
```

- `choice` (string) and `type` are required; `confidence` and `probabilities` are
  optional in the schema. All examples (API reference, tutorial, both blog posts)
  include a probability for every option.
- `confidence` describes "how concentrated the distribution is", not whether the
  answer is correct. We should log it but not use it as a probability.
- The key order inside `probabilities` changes between calls. Index by option name.
- An answer can be `{"type": "refusal"}` with no probabilities. Other questions in
  the same request are still answered.
- `scripts/jev_smoke.py` reads only `choice` and `confidence`, which is why item 10's
  note suspected a top-1 answer. The docs say otherwise.

## Pricing

- Billed on input tokens only; output tokens are free.
- $0.042 per million input tokens (blog posts quoting the model page, observed
  2026-09-19). This matches every documented `cost` value exactly:
  476 × 0.042e-6 = 0.000019992, 447 × 0.042e-6 = 0.000018774,
  357 × 0.042e-6 = 0.000014994. The rate is therefore consistent, but it is still
  vendor data, `TODO(verify)` on the model page.
- The docs disagree on the unit of `cost`: the API reference says "credits", the
  tutorial says USD.
- Every request carries the full criteria text, so input tokens grow with the number
  and length of options.

Rough budget, assuming 500 input tokens per decision (state, briefing, 6–8 options
with short criteria; to be measured):

| Run | Decisions | Cost |
|---|---|---|
| 1 decision | 1 | $0.000021 |
| 50 animals × 200 steps | 10,000 | $0.21 |
| same, 5 seeds | 50,000 | $1.05 |

Cost is not the constraint. Latency and rate limits are.

## Limits, latency, batching

| Item | Finding |
|---|---|
| Questions per request | "As many as you want" (blog). Answered in parallel; they cannot see each other's answers. LiteLLM's proxy caps at 128, which is LiteLLM's limit, not Jev's |
| Options per choice | Not stated for Jev. LiteLLM validates 2–255 for this route. `TODO(verify)`: try 8 options, which is our maximum |
| States per request | One. `state` may be an object or array, but that is still one context |
| Rate limits | Not stated. Only HTTP 429 is documented. `TODO(verify)` |
| Latency | p50 0.194 s, p95 0.63–0.69 s on 40–60 requests (one OpenRouter blog run, 2026-09-19) |
| Streaming | Not available in the schema |

**Why not put many animals into one request?** Several questions can share one
`state`, so one could put every animal's perceptions into a single state and ask one
question per animal. That breaks two things: each animal would get other animals'
perceptions in its context, and TypeSafe says Jev loses accuracy with irrelevant
detail in the state. One request per animal, sent concurrently, keeps the question
equal to what the Decider student sees.

At 0.2 s per request, 50 animals per step are about 10 s per step sequentially and
well under a second with 16–32 parallel requests, if rate limits allow it.

## Errors

Documented status codes: 400 (bad request), 401, 402 (out of credits), 403, 404,
413 (payload too large), 429 (rate limit), 500, 502 and 529 (provider error), 503,
524 (timeout). The body carries `error.code`, `error.message`, `error.metadata`.

For item 9: retry with backoff on 429, 5xx, 524, 529; fail fast on 400, 401, 402,
403, 413; treat a `refusal` answer or missing or unknown option names in
`probabilities` as an error (our rule: a missing probability is never a silent zero).

## Calibration and determinism

- TypeSafe says Jev is calibrated "in aggregate": stated 0.8 means about 80 %
  correct across many answers. No ECE number is published. `TODO(verify)` on our own
  data; that is what our ECE metric is for.
- Probabilities vary slightly between identical calls (a `noul` value of 0.42 and
  then 0.44 in one blog example; variation on `choice` is untested). No seed
  exists. So:
  - the disk cache in item 9 is what makes a run reproducible, and cached decisions
    must be marked `cached=True`;
  - the brain's contract test (deterministic under a fixed seed) runs against the
    fake backend, as planned.
- Option-order sensitivity is not documented. The tutorial warns that rewording one
  option shifts probabilities on the others. We keep shuffling in `encoding/`.

## What this means for item 9 (Jev brain) and item 26 (distillation)

1. **Read `probabilities`, not `choice`.** Map by option name back to
   `Question.options`. Check that the key set equals the offered options exactly.
2. **Renormalise.** Values are probably rounded to two decimals, so they may not
   sum to 1 within the ±1e-6 our `Decision` contract requires. Renormalising inside
   the brain is a brain detail, not a contract change. Log the raw values too.
3. **Zeros in the teacher.** Our KL already clips at eps=1e-12, so it stays finite,
   but a teacher zero means the student is never penalised for mass on that option.
   If rounding is confirmed, item 26 should consider light smoothing of the teacher
   distribution, and the write-up should state the resolution.
4. **Version pinning.** Send `typesafe/jev-1.13` (as the smoke script does) and
   store the returned dated snapshot (e.g. `typesafe/jev-1.13-20260917`) as
   `Decision.model_version`. The `jev-1.13.0` in CLAUDE.md is only illustrative;
   the real ids look like the ones above. Never use `~typesafe/jev-latest` in
   experiments. `TODO(verify)`: whether a dated snapshot stays served once a newer
   1.13 snapshot appears (the docs only say the pinned id "resolves to the current
   1.13 release").
5. **Question text and briefing.** `Question` already carries `briefing`
   (`encoding/question.py`), so where the briefing comes from is settled. How the Jev
   brain renders it (inside `state`, e.g. as an object with perceptions and briefing,
   or inside `instructions`) is a brain detail for item 9. The neutral question line
   for Jev's `instructions` field is still open, as for Decider in
   [decider-2b.md](decider-2b.md). Note that its "Open contract question" section
   predates the `briefing` field and says `Question` has only `state_text` and
   `options`, which is no longer true. Both brains must see the same wording.
6. **Option criteria.** `criteria` values may be `null`. Short neutral descriptions
   ("Step one cell north.") may help Jev, but they are prompt text and must contain
   no advice. To decide in item 9.
7. **Parallel calls.** One request per animal, a bounded thread pool, retry with
   backoff. The concurrency limit goes into config once rate limits are known.

## Open questions / TODO(verify)

Run `scripts/jev_smoke.py` with a key (it now prints the full `probabilities`
object; it is a manual script and the one place besides `brains/jev.py` that calls
the network, never run by tests) and check:

- [ ] `probabilities` is present for every option on every call.
- [ ] Values have two decimals, or more.
- [ ] Sum of `probabilities` (does it equal 1 exactly?).
- [ ] An 8-option choice with our option labels (spaces in keys, e.g. `move north 2`).
- [ ] Same request twice: size of the run-to-run variation.
- [ ] Same request with options reversed: order sensitivity.
- [ ] The rate limit we hit with 16–32 parallel requests.
- [ ] Our own latency (vendor blog: p50 0.194 s, p95 0.63–0.69 s on one small run).
- [ ] Our own input-token count per decision (the budget assumes 500).
- [ ] Vendor statements taken on trust so far: output tokens are free, questions
      in one request are answered independently, calibration "in aggregate",
      jev-1.14 release date.
- [ ] Price per million input tokens on the Jev model page.
- [ ] Whether to move to `typesafe/jev-1.14` before the teacher logs are collected
      (team decision; 1.13 is still served).

## Sources

- [OpenRouter API reference: Submit a Decisions request](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [OpenRouter Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial)
- [OpenRouter blog: What is Jev?](https://openrouter.ai/blog/insights/what-is-jev/)
- [OpenRouter blog: Jev vs LLM](https://openrouter.ai/blog/tutorials/jev-vs-llm-when-to-use-each/)
- [LiteLLM: Decisions](https://docs.litellm.ai/docs/decisions) (proxy limits, not Jev limits)
- TypeSafe docs cited by the pages above but not read: docs.typesafe.ai/models,
  docs.typesafe.ai/confidence, docs.typesafe.ai/concepts/system-one
