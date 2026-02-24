#!/usr/bin/env python3
"""Project Titan -- PostgreSQL database setup script.

Connects to PostgreSQL, executes the ``init_db.sql`` schema file, and
verifies that all expected tables were created successfully.  Designed
to be run as a standalone script during initial deployment or when the
database needs to be re-initialized.

Usage::

    python scripts/setup_db.py

The script reads connection parameters from the same environment
variables (and ``.env`` file) used by the main application via
``config.settings``.  It can also be configured directly through the
standard ``POSTGRES_*`` environment variables.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import psycopg2
from psycopg2 import OperationalError
from psycopg2 import errors as pg_errors

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_INIT_SQL_PATH = _SCRIPT_DIR / "init_db.sql"

# Ensure project root is on sys.path so ``config.settings`` can be imported
# regardless of where the script is invoked from.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_SEC = 2.0
RETRY_BACKOFF_MULTIPLIER = 2.0

# Every table that init_db.sql is expected to create.  Used for the
# post-execution verification step.
EXPECTED_TABLES = [
    "trades",
    "trade_legs",
    "account_snapshots",
    "circuit_breaker_state",
    "model_versions",
    "agent_decisions",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_connection_params() -> dict[str, str | int]:
    """Load PostgreSQL connection parameters from config/settings.py.

    Falls back to sensible defaults if the settings module cannot be
    imported (e.g. missing pydantic-settings in a minimal environment).
    """
    try:
        from config.settings import get_settings

        settings = get_settings()
        return {
            "host": settings.postgres.host,
            "port": settings.postgres.port,
            "dbname": settings.postgres.db,
            "user": settings.postgres.user,
            "password": settings.postgres.password.get_secret_value(),
        }
    except Exception as exc:
        print(f"[WARN] Could not load config.settings ({exc}); using env var defaults")
        import os

        return {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": int(os.environ.get("POSTGRES_PORT", "5432")),
            "dbname": os.environ.get("POSTGRES_DB", "titan"),
            "user": os.environ.get("POSTGRES_USER", "titan"),
            "password": os.environ.get("POSTGRES_PASSWORD", ""),
        }


def _connect_with_retries(
    params: dict[str, str | int],
) -> psycopg2.extensions.connection:
    """Attempt to connect to PostgreSQL with exponential backoff.

    Retries up to ``MAX_RETRIES`` times. This is useful when the script
    is executed before the PostgreSQL container is fully ready (e.g.
    during ``docker compose up``).

    Returns a live ``psycopg2`` connection on success.
    Raises ``SystemExit`` if all retries are exhausted.
    """
    delay = INITIAL_RETRY_DELAY_SEC

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = psycopg2.connect(**params)
            conn.autocommit = True
            print(
                f"[OK]   Connected to PostgreSQL at "
                f"{params['host']}:{params['port']}/{params['dbname']} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            return conn
        except OperationalError as exc:
            if attempt < MAX_RETRIES:
                print(
                    f"[WAIT] Connection attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
                delay *= RETRY_BACKOFF_MULTIPLIER
            else:
                print(
                    f"[FAIL] All {MAX_RETRIES} connection attempts exhausted. "
                    f"Last error: {exc}"
                )
                sys.exit(1)

    # Unreachable, but satisfies type checkers.
    sys.exit(1)


def _read_sql_file(path: Path) -> str:
    """Read and return the contents of an SQL file.

    Raises ``SystemExit`` if the file does not exist.
    """
    if not path.exists():
        print(f"[FAIL] SQL file not found: {path}")
        sys.exit(1)

    sql = path.read_text(encoding="utf-8")
    print(f"[OK]   Loaded SQL schema from {path} ({len(sql):,} bytes)")
    return sql


def _execute_schema(conn: psycopg2.extensions.connection, sql: str) -> None:
    """Execute the full SQL schema against the database.

    Uses a single cursor to run the entire script. Individual statements
    that fail because the object already exists (e.g. ``CREATE TABLE``
    on re-run) are caught so the script is idempotent.
    """
    print("[INFO] Executing schema...")
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        print("[OK]   Schema executed successfully")
    except pg_errors.DuplicateTable as exc:
        # Tables already exist -- this is fine on subsequent runs.
        print(
            f"[WARN] Some objects already exist "
            f"(safe to ignore): {exc.diag.message_primary}"
        )
    except Exception as exc:
        print(f"[FAIL] Schema execution failed: {exc}")
        sys.exit(1)


def _verify_tables(conn: psycopg2.extensions.connection) -> bool:
    """Verify that all expected tables exist in the public schema.

    Returns ``True`` if every table in ``EXPECTED_TABLES`` is present.
    """
    print("[INFO] Verifying tables...")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        existing_tables = {row[0] for row in cur.fetchall()}

    missing: list[str] = []
    for table in EXPECTED_TABLES:
        if table in existing_tables:
            print(f"  [OK]   {table}")
        else:
            print(f"  [MISS] {table}")
            missing.append(table)

    if missing:
        print(f"[FAIL] Missing tables: {', '.join(missing)}")
        return False

    print(f"[OK]   All {len(EXPECTED_TABLES)} expected tables verified")
    return True


def _verify_initial_data(conn: psycopg2.extensions.connection) -> None:
    """Check that seed data (circuit breaker initial state) is present."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM circuit_breaker_state;")
        count = cur.fetchone()[0]

    if count >= 1:
        print(
            f"[OK]   circuit_breaker_state has {count} row(s)"
            f" (initial seed data present)"
        )
    else:
        print(
            "[WARN] circuit_breaker_state is empty"
            " -- initial seed row may not have been inserted"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full database setup workflow.

    Steps:
        1. Load connection parameters from config/settings or env vars.
        2. Connect to PostgreSQL with retries (exponential backoff).
        3. Read ``scripts/init_db.sql``.
        4. Execute the schema.
        5. Verify all expected tables exist.
        6. Verify seed data is present.
    """
    print("=" * 60)
    print("  Project Titan -- PostgreSQL Database Setup")
    print("=" * 60)
    print()

    # 1. Load connection parameters
    params = _load_connection_params()
    safe_display = (
        f"{params['user']}@{params['host']}:{params['port']}/{params['dbname']}"
    )
    print(f"[INFO] Target: {safe_display}")
    print()

    # 2. Connect with retries
    conn = _connect_with_retries(params)

    try:
        # 3. Read the SQL file
        sql = _read_sql_file(_INIT_SQL_PATH)
        print()

        # 4. Execute schema
        _execute_schema(conn, sql)
        print()

        # 5. Verify tables
        tables_ok = _verify_tables(conn)
        print()

        # 6. Verify seed data
        _verify_initial_data(conn)
        print()

        # Final summary
        print("=" * 60)
        if tables_ok:
            print("  Database setup completed successfully.")
        else:
            print("  Database setup completed with warnings -- see above.")
        print("=" * 60)

        if not tables_ok:
            sys.exit(1)

    finally:
        conn.close()
        print("[INFO] Connection closed.")


if __name__ == "__main__":
    main()
