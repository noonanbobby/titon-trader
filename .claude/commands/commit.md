Review all staged changes with `git diff --cached`. If nothing is staged, review
unstaged changes with `git diff` and stage the relevant files.

Write a conventional commit message using the format:
  type(scope): description

Where type is one of: feat, fix, refactor, test, docs, chore, perf, ci, build

Run `uv run ruff check --fix . && uv run ruff format .` before committing.
Run `uv run pytest` to verify tests pass. If tests fail, fix the issues first.
