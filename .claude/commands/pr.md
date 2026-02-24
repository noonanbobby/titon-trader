---
argument-hint: [branch-name]
description: Create a pull request
---
Create a PR from the current branch to main. Steps:

1. Run the full test suite: `uv run pytest`
2. Run linting: `uv run ruff check . && uv run ruff format --check .`
3. Summarize ALL changes since branching from main (not just the latest commit)
4. List affected files grouped by component (broker, strategies, signals, risk, ai, ml, etc.)
5. Note any breaking changes or new dependencies
6. Create the PR with `gh pr create`
