# Experiment repository: reversible edits

- This directory is an independent Git repository. Run Git commands here, not in the parent workspace repository.
- Before the first edit in each user-requested task, run `git checkpoint "before-<short-task-name>"`. It saves tracked and non-ignored untracked working files without changing `main`, HEAD, or the index. Stop editing if the checkpoint fails.
- Use `apply_patch`, preserve unrelated changes, and run tests relevant to the requested change.
- User preference (2026-10-10): after completing each task's edits and relevant verification, automatically commit that task's changes in this repository without asking again. Stage only task-owned files or hunks; leave unrelated pre-existing changes uncommitted. Do not push unless explicitly requested. Report incomplete or failed verification rather than claiming completion.
- Results, checkpoints and raw datasets remain ignored and outside this recovery mechanism. Changes to those assets require their own scoped backup.
- `git recover` previews restoration to the latest checkpoint; `git recover --apply` saves a rescue checkpoint and restores working files. HEAD and the index stay unchanged.
- The full recovery guide is `../../GIT_RECOVERY.md`. Use the current source and protocol rather than historical results in the parent overview.
