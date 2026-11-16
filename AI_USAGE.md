# AI usage record

**Purpose.** Work item 7 of the assignment asks for a chronological record of
the AI tools used, with prompts and responses.

**How this file was produced.** It is written by the AI assistant from the
session record of this project. User prompts are reproduced verbatim where the
session still holds the exact text. Where a prompt predates a context-compaction
boundary, the entry is marked `paraphrase` and the wording is not presented as
the user's own. Responses are summarised, not reproduced; the full transcripts
are held in the Hermes session store and can be exported on request.

**Model identification.** The session metadata reports the model per turn. The
labels below are transcribed from that metadata. Earlier turns ran under a model
whose identifier is no longer in the active context; those entries say
`model not resolved` rather than guessing.

---

## 2026-11-13 — Day 1: data audit and preprocessing

**Model:** not resolved in the active context.
**Request (paraphrase):** build a reproducible system for the assignment: read
the four L1C products, measure radiometry, registration, water and normalisation,
and keep a plan.

**Produced:** project scaffold, `data/safeloader.py` with the PB 04.00
radiometric conversion, `data/register.py` (ECC, sign convention corrected and
self-tested), `preprocess/normalize.py`, `preprocess/water.py`,
`configs/data_manifest.yaml`, `PLAN.md`, `outputs/audit/data_audit.md`.

**Corrections made during this phase:** band-name mapping `B1`/`B01`; the ECC
sign convention, verified with a synthetic shift; the water threshold moved from
0.2 to the McFeeters 0.0 boundary after measuring 30 percentage points of
instability between dates; NDWI returning 0.0 on a zero denominator changed to
NaN.

## 2026-11-14 — Day 2: public data, classical baseline, labels, patches

**Model:** not resolved in the active context.
**Request (verbatim, recovered from the session record):**
> "Proceed with implementing OSCD loader & label tooling. Implement that, verify
> if it's actually done & you're not just hallucinating & then report back. Does
> D2.6 partitions, D2.5 label set(where you build the tooling & I make the
> calls), D2.7 patches + index (this requires D2.5 & D2.6). Complete all of this,
> then we'll move onto D2.4 OSM (https://overpass-api.de/). Go step-by-step,
> follow a to-do list. Make sure that you don't break/corrupt the already
> existing project's features/files/folders. Fetch & refer to official resources
> via the web, wherever required, don't act as if you know everything. Read the
> files that you'll modify/update before editing them, don't just blindly make
> the modifications."

**Produced:** OSCD download and audit, `data/oscd_original.py`,
`configs/oscd_region_splits.yaml`, `scripts/prepare_local_partitions.py`,
`data/local_patches.py`, `train_local_manual.py`,
`scripts/rasterize_manual_labels.py`, `scripts/export_review_candidates.py`,
`tests/test_day2_safety_guards.py`.

**Failures recorded rather than hidden:** the two delegated read-only code
reviews returned HTTP 401 from the provider, so the review was done manually; OSM
/ Overpass was unreachable from this machine (HTTP 406 then network errors), so
D2.4 has no results.

## 2026-11-14 — Independent audit response, P0 validity and registration

**Model:** reported as `deepseek-flash` in the session metadata at the start of
this phase.

**Request (verbatim):**
> "I am providing an independent repository audit. Read the complete Markdown
> file before making any changes. Do not attempt to fix every issue at once.
> Follow the priorities and acceptance criteria in the audit. **Start with the
> recommended first task:** Implement the paired observation-validity pipeline
> … Do not fabricate results or mark untested functionality as complete. Work on
> this first task only, then stop and report."

**Produced:** `data/grids.py`, `data/validity.py`, strict quality-mask readers,
masked ECC, the single reference-scaled score, validity integration in the
baseline, partition generator and annotation rasterizer, and
`tests/test_observation_validity.py`.

**Follow-up request (verbatim):**
> "1. Complete mask-aware registration … 2. Rebuild the spatial partitions …
> 3. Regenerate candidate review layers … 4. Resolve the water and
> annotation-validity rules … 5. Final verification … Stop after this work."

**Produced:** mask-aware `run_registration.py` with an independent control-window
check, `outputs/partitions_v2_observation_valid/`,
`outputs/audit/v2_observation_valid/`, `outputs/labels/v2_observation_valid/`,
the water/validity policy fix in the schema, docs and rasterizer, and
`docs/reviews/p0_validity_registration.diff`.

## 2026-11-14 — Reporting corrections

**Request (verbatim, abridged):** the user corrected an area figure
(2,809,724 px at 20 m is 1,123.9 km², not 281 km²), asked for the relationship
between the three candidate thresholds, asked what "usable" meant for a split,
and asked to confirm test-label exclusion.

**Produced:** corrected arithmetic, an explicit table of which threshold
produced which output, exact per-orientation counts, and a new unit test that
proves the trainer refuses an index which read test labels. No pipeline
parameter was changed.

## 2026-11-14 — Unique-pixel metrics, normalisation run, inference, deliverables

**Request (verbatim, abridged):** "I am starting the manual annotation workflow
in QGIS. While I work on the labels, continue with tasks that do not depend on
the human annotations." Four priorities: unique-pixel statistics, the
normalisation run, georeferenced inference, and the final deliverables. "Do not
create labels, invent detections, start local supervised training, or use
unreviewed candidate masks as ground truth."

**Produced:** `utils/unique_metrics.py`, `tests/test_unique_metrics.py`,
`scripts/run_normalization.py` executed end to end into
`outputs/audit/normalization_v2_observation_valid/`,
`scripts/predict_safe_pair.py` and a 1024 x 1024 smoke run,
`configs/detection_register_schema.yaml`, `docs/detection_register.md`,
`docs/onboard_architecture.md`, `docs/reproducibility.md`,
`scripts/make_run_manifest.py`, `tests/test_inference_contract.py`, and this
report draft.

**Finding that changed a documented claim:** on the corrected validity and score
basis, a global linear normalisation gives the largest stable-land error
reduction for both pairs (39.6 % and 32.0 %), not histogram matching (36.5 % and
31.5 %). `config.yaml` still states histogram. The value was left unchanged and
the conflict is recorded as open item 1 in `docs/report.md`.

---

## Standing instructions from the user, applied throughout

- Do not fabricate outputs, metrics, labels or timings.
- Do not convert candidate scores or OSM records into ground truth.
- Keep the official OSCD test labels out of training and model selection.
- Never overwrite an existing generated output.
- Keep the work reproducible: record the commit, configuration, inputs, seeds
  and outputs.
- Report blockers instead of working around them silently.
- Use Simplified Technical English in reports.

## Tooling used

- Hermes Agent, with browser/web extraction for the official sources cited in
  `PLAN.md` section 12.
- Python 3.11 in the conda `gpu` environment: torch 2.11.0+cu128, rasterio
  1.4.4, numpy, scipy, opencv 4.13.0, matplotlib, shapely, pyproj.
- Git, local only. Nothing has been pushed from this machine without an
  explicit request.

## Gaps in this record

- Full prompt and response text for the Day 1 and early Day 2 turns is not in
  the active context. Those entries are summaries.
- Two turns are marked `model not resolved`; the identifier was not preserved
  across a context-compaction boundary.
- No AI tool other than this assistant wrote project files.

To complete the record for submission, export the full session transcript and
append it as an annex, keeping this file as the index.
