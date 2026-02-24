---
argument-hint: [test-path-or-pattern]
description: Run tests with coverage
---
Run the test suite with pytest. If an argument is provided, run only matching tests.
Otherwise run the full suite.

Commands:
- Full suite: `uv run pytest -v`
- Specific file: `uv run pytest -v $ARGUMENTS`
- With coverage: `uv run pytest --cov=src --cov-report=term-missing -v`

Report any failures with the relevant file path and line number.
