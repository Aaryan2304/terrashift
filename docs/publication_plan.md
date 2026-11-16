# Publication plan

**Status: plan only. Nothing in this document has been executed.**

The local and remote histories share no ancestor. Local `main` is the original
development line; `origin/main` is a single re-rooted "Initial Commit" pushed for
the assignment submission. Neither can be merged or pushed onto the other
without a history operation, and history operations are not permitted.

Measured state on 2026-11-15:

| Ref | Commit | Note |
|---|---|---|
| `main` | `bbb2550` "Guard local index and candidate outputs" | development line, 27 modified and 37 untracked files in the worktree |
| `terrashift-initial` | `e2eaa6c` "Initial Commit" | tracks `origin/main`, 56 files |
| `origin/main` | `e2eaa6c` | same commit |

`git merge-base HEAD origin/main` returns nothing: unrelated histories.

## What must not happen

- No force push, to any branch.
- No reset, rebase or amend of published commits.
- No merge of the two histories.
- No staged or committed state without a decision.
- No deletion of historical outputs or checkpoints. They are gitignored, so
  they are not at risk from either option.

## Recommended route: new branch from the published root

This adds work as an ordinary fast-forward from `e2eaa6c`, which needs no force
push, keeps local `main` intact, and leaves the published history linear.

    # 1. Start from the published root, in a separate worktree so the working
    #    tree under review is never disturbed.
    git worktree add ../terrashift-submission terrashift-initial

    # 2. Apply the reviewed tree. The deliverable is the working tree, not the
    #    development commits, so a single reviewed commit is the honest unit.
    cd ../terrashift-submission
    rsync -a --exclude=.git --exclude=outputs --exclude=data \
          ~/projects/skyserve_assignment/ ./
    git add -A
    git status --short          # must contain no data, rasters, checkpoints, labels
    git commit -m "Add observation validity, training, evaluation and quantisation"

    # 3. Verify, then push as a normal fast-forward.
    git push origin HEAD:refs/heads/submission

    # 4. Open a pull request submission -> main. Merge is the owner's decision.

Local `main` is never touched. `terrashift-initial` stays where it is.

## Alternative route: rebuild the root

Only if a single-commit submission is required again. This rewrites the public
branch, so it needs a new branch name, not `main`:

    git checkout --orphan submission-2026-11-15
    git add -A && git commit -m "TerraShift submission"
    git push origin HEAD:refs/heads/submission-2026-11-15

The existing `main` on the remote is left alone. Deleting or replacing it is a
remote history rewrite and is out of scope.

## Pre-flight checks, all mandatory

Run in the review worktree before any push.

1. `git status --short` shows only intended source, documentation and test files.
2. No imagery, rasters or model files are tracked:
   `git ls-files | grep -icE '\.(tif|jp2|SAFE|pt|onnx|pbf)$'` must return 0.
3. No file over 10 MB: `git ls-files -z | xargs -0 ls -l | awk '$5 > 10000000'`.
4. No manual-label GeoJSON: `git ls-files 'labels/*'` shows only the template.
5. No credentials: grep the diff for `api_key`, `token`, `password`, `secret`,
   `PRIVATE KEY`, and for the known provider key patterns.
6. `SENTINEL_DATA_ROOT` substitution holds: no absolute `/home/aaryan/...` path
   in the new commit.
7. Full test suite passes: `PYTHONPATH=. python -m unittest discover -s tests`.
8. `AI_USAGE.md` is current, and states which results are OSCD development
   results and which are local.
9. README commands work from a clean clone with only `SENTINEL_DATA_ROOT` set.

### Checked on 2026-11-15 against the current state

| Check | Result |
|---|---|
| `outputs/` in the published root `e2eaa6c` | absent |
| `.tif` / `.jp2` / `.SAFE` / `.pt` / `.onnx` / `.pbf` in the published root | absent |
| Tracked files over 10 MB | none |
| Tracked label files | `labels/manual_labels.template.geojson` only |
| Secret patterns in the worktree diff | none |
| Absolute `/home/aaryan` paths outside `outputs/` | none |

Two files in the **local** history, `outputs/audit/offset_check.txt` and
`outputs/train_oscd_baseline_v1.txt`, contain absolute paths. They were committed
in `44d89fd`, before `.gitignore` gained the `outputs/` rule, and they are absent
from the published root. Both routes below exclude `outputs/`, so they cannot
reappear. They are not deleted from local history, because that would rewrite it.

## Open decisions for the owner

1. **Licence.** The repository has none. A public repository with no licence
   grants no rights. Options: MIT or Apache-2.0 for the code, with a clear note
   that the OSCD labels are CC BY-NC-SA and the imagery is modified Copernicus
   data. No licence file will be added without a decision.
2. **Route.** New branch and pull request, or a rebuilt single-commit root.
3. **Commit shape.** One reviewed commit, or a short series that separates the
   validity chain, training and evaluation, quantisation, and cleanup.
4. **Report form.** `docs/report.md` is the source; a PDF is built separately and
   is not tracked.

## Rollback

Both routes only add a branch. Until a pull request is merged, `main` on the
remote is unchanged. Deleting the new branch restores the previous state
exactly, with no history operation on any published ref.
