# 008 — Publication goes out as a new branch from the published root

**Status:** proposed, awaiting the owner's decision
**Date:** 2026-11-15

## Context

The local development history and the branch published on GitHub share no
ancestor. `git merge-base HEAD origin/main` returns nothing.

- local `main` is at `bbb2550` "Guard local index and candidate outputs";
- `terrashift-initial` and `origin/main` are at `e2eaa6c` "Initial Commit".

The working tree holds the whole validity chain, the training, evaluation and
quantisation work, and the cleanup, none of it committed.

## Decision proposed

Add the work as an ordinary fast-forward from `e2eaa6c` on a new branch, in a
separate worktree, then open a pull request. Local `main` is never touched, no
published commit is rewritten, and no force push happens.

## Alternatives rejected

- **Force push local history over the remote.** Destroys the published commit and
  is explicitly out of scope.
- **Merge the two histories.** They are unrelated; a merge commit would claim a
  relationship that does not exist.
- **Rebuild the root on `main`.** Rewrites a published branch. A rebuilt root is
  acceptable only on a new branch name.

## Consequence

Two decisions remain with the owner: the route, and the licence. The repository
has no licence file, and a public repository without one grants no rights.

Until a pull request is merged, deleting the new branch restores the previous
state exactly.

## Evidence

`docs/publication_plan.md`, which lists the pre-flight checks and their measured
results: no `outputs/` or raster files in the published root, no tracked file over
10 MB, no secret patterns, no absolute home paths outside `outputs/`.
