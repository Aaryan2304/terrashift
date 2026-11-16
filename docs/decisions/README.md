# Decision records

One file per technical decision that changed the shape of the system. Each record
states the context, the decision, the alternatives that were rejected and why,
the consequence, and the evidence in the repository.

The full chronological log is in `PLAN.md` section 10. These records are the long
form of the entries that carry real engineering weight.

| # | Decision | Status |
|---|---|---|
| 001 | Observation validity and water are separate concepts | accepted |
| 002 | Registration is estimated on valid pixels only | accepted |
| 003 | The percentile threshold is a prior, not an operating point | accepted |
| 004 | Validation counts each reviewed pixel once | accepted |
| 005 | Original OSCD is the benchmark; the official test split is opened once | accepted |
| 006 | GPU training configuration on a 4 GB laptop GPU | accepted |
| 007 | Static INT8 QDQ, measured on the provider it will run on | accepted |
| 008 | Publication goes out as a new branch from the published root | proposed |
| 009 | ONNX Runtime's CPU QDQ optimiser corrupts this graph | diagnosed, correction proposed |

Decisions still open with the owner:

- normalisation method: histogram (change preservation on the positive controls)
  against global (lower stable-land error). See `docs/report.md` section 8.
- the software licence. The repository has none, so a public copy grants no
  rights.
- a test-set INT8 evaluation, if the submission needs one. That would be a second
  deliberate use of the locked split.
