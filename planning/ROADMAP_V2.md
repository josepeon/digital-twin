# Digital Twin — Roadmap v2.0

> Forward-looking improvement plan. `planning/DIGITAL_TWIN_PLAN.md` covers
> the original Phases 1–7 (✅ all done) and outlines Phase 8 future
> extensions. This document picks up where Phase 8 leaves off, scoped to
> **no-new-paid-services**: existing ElevenLabs / Groq / Whisper usage
> continues, but anything that introduces a new monthly bill is deferred.

## Context

The twin is in genuinely good shape: deployed on Railway, dual-voice
cloned, RAG-balanced across sources, A- on the LLM-as-judge style eval,
46/46 on identity eval, PWA-installable. 44 unit tests passing.

The gaps now are:

- **Visibility** — no per-run cost tracking; manual eval; no production metrics
- **Reuse** — twin has its own everything; SIA already ships canonical
  implementations of cost tracking, reproducibility, metrics. Same dedup
  opportunity we just landed for `multi_agent_planner`.
- **Test coverage** — `twin_api.py`, `voice.py`, `memory.py` have zero
  unit tests. Only parsers/PII/personality_engine/models are covered.
- **The free items in Phase 8** — real-time learning, multi-persona,
  local LoRA — never started. All free; ready to ship.
- **Connecting the OH AI loop** — twin captures real conversations; SIA
  trains models. Wiring them together turns usage into training data.

---

## Cost constraint

This roadmap excludes anything that would add a *new* paid service:

| Phase 8 item | Status here | Why |
|---|---|---|
| 8a Phone Call Agent (Twilio) | ⛔ Deferred | $1/mo + $0.013/min Twilio billing |
| 8e Video Avatar (HeyGen/Synthesia) | ⛔ Deferred | Monthly avatar service subscription |
| 8d Autonomous Agent — sending side | ⛔ Deferred | Each sent reply spends ElevenLabs / could trigger spam-flag billing |
| 8d Autonomous Agent — draft/approval queue | ✅ Included | Read-only — Gmail/IG APIs free tier, no sends |
| Everything else (8b, 8c, 8f, 8g) | ✅ Included | Pure code; free APIs only |

When the budget cap relaxes, the deferred items become a Sprint 6 add-on.

---

## Bugs / honesty

No glaring bugs surfaced. README is accurate. The main weakness is
**test coverage on the runtime path** — `twin_api.py` (409 LOC),
`voice.py` (209 LOC), `memory.py` (178 LOC) are uncovered. Smoke-tested
in production but not in CI.

---

## Tier 1 — Transformational

### T1.1. Cross-repo: reuse SIA's `cost_tracker`, `reproducibility`, `metrics`
**Problem:** The twin makes Groq + ElevenLabs + Whisper calls every
request. No per-run accounting. No process metrics. Eval scripts use
default randomness so two runs disagree slightly.

**Plan:** install SIA via the existing pattern (`pip install -e '.[sia]'`
in MAP works the same here). Then:
- Replace any twin-local cost helpers with `from
  self_improving_agent.observability.cost_tracker import attribute,
  begin_run, record_usage`. The twin doesn't ship its own cost tracker
  today — this is purely additive: wrap each Groq/ElevenLabs/Whisper
  call with `record_usage(...)` so the existing dashboard pattern
  works.
- Pin training-related randomness via `from
  self_improving_agent.reproducibility import set_seed`.
- Add a `/metrics` Prometheus endpoint by importing `make_sia_registry`
  from `self_improving_agent.observability.metrics`.

**Effort:** Small (1 day).
**Files:** `pyproject.toml` (add `[sia]` extra), `src/personality_engine.py`,
`src/voice.py`, `src/twin_api.py` (one new endpoint), small wrapper module.

### T1.2. Twin → SIA distillation bridge
**Problem:** The twin already captures every real conversation in
SQLite memory. SIA already has a distillation pipeline (`scripts/
distill_from_groq.py`) that takes prompts + responses and produces SFT
training data. Wiring them connects "twin gets used in production" to
"the next twin version gets better." Same closed-loop pattern we built
for MAP.

**Plan:**
- New `scripts/export_to_sia.py` (mirrors MAP's). Reads
  `data/personal/memory.db`, extracts (user_msg, twin_response) pairs,
  filters by quality (length, no error markers), writes SIA-format JSONL.
- Optional `--run distill` flag invokes SIA's `run_distillation()` on the
  exported rows when the `[sia]` extra is installed.
- Result: each week of live usage produces a new training set the local
  LoRA fine-tune can consume.

**Effort:** Small (1 day).
**Files:** new `scripts/export_to_sia.py`.

### T1.3. CI eval gating
**Problem:** `tests/eval_identity.py` runs 46 identity questions + style
eval, but it's manual. A bad RAG rebalance or system prompt change can
ship without anyone noticing.

**Plan:**
- New `.github/workflows/ci.yml` running unit tests on every push.
- Second job `eval-identity` that runs the eval against the *remote*
  Groq provider (free tier) on every push to `main`. Soft-gated at
  first; flip to hard gate once a baseline is established.
- Cache the ChromaDB index to keep the job under 60s.

**Effort:** Small (½ day).
**Files:** new `.github/workflows/ci.yml`.

---

## Tier 2 — High-impact additions

### T2.1. SSE streaming on `/ask-jose`
**Problem:** WebSocket clients get streaming today (`/ws`). REST clients
hitting `/ask-jose` get full responses only. The twin's web UI uses WS,
but third-party integrations (CLIs, future OH services) might want SSE
without the WS handshake complexity.

**Plan:** Add streaming via Server-Sent Events on `/ask-jose/stream-sse`
(separate path from the existing audio-stream endpoint). Token-by-token
emission via the existing Groq streaming response.

**Effort:** Small (½ day).
**Files:** `src/twin_api.py`.

### T2.2. Real-time learning (Phase 8b)
**Problem:** Twin's RAG was built from a one-time export of messages.
Recent conversations don't make it back into the index until someone
re-runs the parsers manually.

**Plan:**
- `scripts/ingest_new_data.py` — watch a configured directory for new
  source exports (Instagram dump, ChatGPT export, etc.) and
  incrementally add new messages to ChromaDB without rebuilding from
  scratch.
- Track last-ingest timestamps per source so only new rows process.
- Optional cron via Railway scheduled jobs (free tier).

**Effort:** Medium (1–2 days).
**Files:** new `scripts/ingest_new_data.py`, small helper for incremental
ChromaDB upsert.

### T2.3. Test coverage for runtime code
**Problem:** Three of the runtime modules — `twin_api.py`, `voice.py`,
`memory.py` — have no unit tests. Production smoke-tested, but
refactors are scary.

**Plan:**
- `tests/test_twin_api.py` — FastAPI test client; cover health, ask-jose
  (with mocked engine), auth, websocket happy path.
- `tests/test_voice.py` — mock the ElevenLabs client; cover language
  routing, voice ID selection, error handling.
- `tests/test_memory.py` — SQLite store: add/get/clear, TTL eviction,
  session isolation, WAL mode survival.

Target: bump the runtime-code coverage from ~0% to >80%.

**Effort:** Medium (2 days).
**Files:** new `tests/test_twin_api.py`, `tests/test_voice.py`,
`tests/test_memory.py`.

### T2.4. Phase 8f — Multi-persona modes
**Problem:** Single voice/register. The twin can't switch between
"Professional Jose for OH meetings" and "Casual Jose for friends" on
demand.

**Plan:**
- System prompt variants per persona: professional / casual / creative
- API parameter `mode: "professional" | "casual" | "creative"`
- RAG filter: when mode is set, weight retrieval toward examples
  matching that register (we already detect register; this inverts the
  detection into a filter)
- UI toggle for mode selection.

**Effort:** Small-Medium (1–2 days).
**Files:** `src/personality_engine.py`, `src/twin_api.py`,
`src/static/index.html`.

---

## Tier 3 — Polish & ops

### T3.1. Phase 8g — Run the LoRA fine-tune (locally, free)
**Problem:** `scripts/finetune_lora.py` was built but never run.
Planning doc estimated $2–3/hr cloud GPU. **Pivot:** SIA already ships
an MLX-based fine-tune pipeline that runs on the M4 Max for free.

**Plan:**
- Convert `scripts/export_training_data.py` output (already in chat
  format) into SIA's expected SFT JSONL via `scripts/export_to_sia.py`
  (built in T1.2).
- Run SIA's `train_oh_model.py`-style command against `jose_train.jsonl`
  on the local M4 Max. ~20 min on Apple Silicon with the existing 8B
  base.
- Register the resulting adapter in SIA's model registry.
- Wire `src/personality_engine.py` to *optionally* use the fine-tuned
  adapter as a fallback when Groq is rate-limited (matches the original
  Phase 7 "offline fallback" intent).

**Effort:** Medium (2–3 days, mostly waiting for training + eval).
**Files:** new `scripts/train_local.py` (thin wrapper around SIA),
`src/personality_engine.py`.

### T3.2. Phase 8c — OH integration (no paid deps)
**Problem:** Twin runs standalone on Railway. OH platform has a
multiplayer Unreal experience where the twin could appear as Jose's
avatar.

**Plan:**
- `src/oh_agent.py` — adapter exposing the PersonalityEngine via OH's
  WebSocket protocol.
- No new services — re-uses the existing twin deployment.
- Spatial presence is OH-side work; SIA-side is just an adapter.

**Effort:** Medium (2 days for the adapter; OH-side integration is
separate).
**Files:** new `src/oh_agent.py`, small route in `src/twin_api.py`.

### T3.3. Phase 8d — Autonomous draft queue (NO sending side)
**Problem:** Twin could draft replies to incoming email/DMs but can't
send without a paid trigger (each send → ElevenLabs spend if voice,
risk of platform flags, etc).

**Plan, free portion only:**
- `src/agent.py` — read Gmail unread, generate draft via twin, queue
  for human approval.
- Approval queue endpoint: `GET /agent/queue`, `POST /agent/approve/{id}`.
- The actual *send* step is deferred until budget allows; for now,
  approving copies the draft to clipboard / shows a "send manually" CTA.

**Effort:** Medium (2–3 days).
**Files:** new `src/agent.py`, queue table in `src/memory.py`.

---

## Recommended phasing

### Sprint 1 (~2 days): "Trust + visibility"
- T1.1 cross-repo reuse (cost tracking, reproducibility, metrics)
- T1.3 CI eval gating

### Sprint 2 (~2 days): "Tested + streamable"
- T2.3 runtime test coverage
- T2.1 SSE streaming

### Sprint 3 (~2 days): "Self-feeding loop"
- T1.2 twin → SIA distillation bridge
- T2.2 real-time learning (auto-ingest)

### Sprint 4 (~2-3 days): "Personas + local model"
- T2.4 multi-persona modes
- T3.1 run the LoRA fine-tune locally via SIA

### Sprint 5 (~2-3 days): "Platform integration"
- T3.2 OH agent adapter
- T3.3 autonomous draft queue (no send)

---

## Total scope

- **Tier 1:** ~2.5 days
- **Tier 2:** ~4.5 days
- **Tier 3:** ~7 days

≈ **2 weeks** of focused work — about half of MAP's scope, reflecting
how much further along the twin already is.

## Cross-repo opportunities

Three items deliberately reuse SIA modules rather than duplicate them:

- **T1.1** uses SIA's `cost_tracker` + `reproducibility` + `metrics`
- **T1.2** invokes SIA's `run_distillation()` to turn real conversations
  into training data
- **T3.1** uses SIA's MLX fine-tuning pipeline on the M4 Max instead of
  paying for cloud GPU

This is the third repo to converge on SIA as the canonical infrastructure
library. The pattern continues.

## Deferred (paid)

When the budget gate opens:

- **8a Twilio Phone Agent** — HIGH priority per the original doc; the
  twin could literally answer Jose's phone in his voice.
- **8e Video Avatar (HeyGen/Synthesia)** — visible Jose avatar in the
  PWA + OH integration.
- **8d Autonomous Agent SEND step** — flip the "draft only" queue to
  actually send, after a few weeks of approval-queue review proves the
  draft quality is high enough.

---

## Implementation status (updated 2026-05-11)

### Shipped ✅

All 12 in-scope items above are landed on `main`. Tests: 44 → 202.
Pushed to `github.com/josepeon/digital-twin`.

| Sprint | Items | Outcome |
|---|---|---|
| 1 — Trust + visibility | T1.1, T1.3 | SIA observability wiring (cost tracker + reproducibility + `/metrics`); GitHub Actions with unit + soft-gated eval-identity |
| 2 — Tested + streamable | T2.3, T2.1 | runtime coverage on twin_api/voice/memory (~0% → high); `/ask-jose/stream-sse` OpenAI-compat endpoint |
| 3 — Self-feeding loop | T1.2, T2.2 | `scripts/export_to_sia.py` bridges memory.db → SIA distillation; `scripts/ingest_new_data.py` watches a drop-zone for incremental RAG updates |
| 4 — Personas + local model | T2.4, T3.1 | default / professional / casual / creative modes; `scripts/train_local.py` runs SIA's MLX fine-tune locally on M-series |
| 5 — Platform integration | T3.2, T3.3 | WS `/oh/agent` route for the OH Unreal experience; SQLite draft queue with `POST /agent/draft|approve|reject` — no send |

One production bug surfaced + fixed: `/ask-jose` previously let engine
exceptions propagate raw. Now wrapped in try/except that logs and
returns 500 with detail.

### Still to do 🔲

#### One-time human actions
- **Add `GROQ_API_KEY` repo secret** — Settings → Secrets at
  https://github.com/josepeon/digital-twin/settings/secrets/actions.
  The CI `eval-identity` job no-ops without it.
  See `.github/CI_SECRETS.md`.
- **Run `scripts/train_local.py` end-to-end.** The wrapper exists, the
  19k-pair Phase 7g export exists, MLX is installed. The training run
  hasn't happened.
  ```bash
  python scripts/train_local.py \
      --data data/training/jose_train.jsonl \
      --version v1.0-twin
  ```
  ~20 min on M3/M4 Max. Evaluate with
  `python -m tests.eval_identity --quick`, then promote from CANARY →
  PRODUCTION in SIA's registry.
- **Wire a reader to populate the draft queue.** `POST /agent/draft` is
  built; nothing currently *feeds* it. Gmail API is free — a tiny
  poller that watches unread + POSTs to `/agent/draft` per message is
  ~50 LOC.
- **Plumb T2.2 parser dispatch.** `ingest_new_data.py` calls
  `parser.parse_file(path)` when present, but the existing
  `scripts/parse_*.py` parsers expose `main()` instead. Adding a
  `parse_file(path)` wrapper to each is ~10 LOC per parser; deferred
  because no new exports are currently queued.

#### Deferred (paid services — waiting on budget gate)
- **8a Twilio Phone Agent** — $1/mo + $0.013/min. Twin answers Jose's
  phone in his voice. Skeleton outlined in `DIGITAL_TWIN_PLAN.md` §12.
- **8e Video Avatar** (HeyGen/Synthesia) — ~$20-30/mo subscription.
  Lip-synced video of Jose speaking the responses.
- **8d Autonomous Agent SEND step** — flip the approve handler from
  informational to dispatching via channel API. Each send risks
  ElevenLabs charges + platform spam-flag billing.

#### Cross-repo follow-ups
- Twin's `MemoryStore` (`src/memory.py`) is more battle-tested than
  SIA's `logging/storage.py`. Candidate for an upstream merge after
  MAP's `core/interaction_log.py` lands in SIA.
