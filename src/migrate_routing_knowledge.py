"""Copy a legacy SQLite routing store into an empty PostgreSQL destination."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

import psycopg
from psycopg import sql

from .routing_knowledge import RoutingKnowledge


def migrate(source: Path, dsn: str) -> None:
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as legacy:
        legacy.execute("BEGIN")
        config_row = legacy.execute("SELECT value FROM config WHERE id = 1").fetchone()
        if config_row is None:
            raise ValueError("The SQLite file has no routing configuration.")
        config = json.loads(config_row[0])
        if config["version"] != 1:
            raise ValueError("Unsupported SQLite routing database version.")

        tables = {name for (name,) in legacy.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "experience" not in tables:
            raise ValueError("The SQLite file has no routing experience table.")
        rows = {
            "experience": legacy.execute("""
                SELECT controller_id, task_type, agent_id, observations, successes, quality_sum, service_seconds
                FROM experience
            """).fetchall(),
            "execution_failures": legacy.execute("""
                SELECT controller_id, task_type, agent_id, failures FROM execution_failures
            """).fetchall() if "execution_failures" in tables else [],
            "evaluated_prompts": legacy.execute(
                "SELECT digest FROM evaluated_prompts"
            ).fetchall() if "evaluated_prompts" in tables else [],
            "published_readiness": [
                (row_id, bool(ready))
                for row_id, ready in legacy.execute("SELECT id, ready FROM published_readiness")
            ] if "published_readiness" in tables else [],
        }

    with closing(RoutingKnowledge(
        dsn, config["scope"], round_size=config["round_size"], mvp_rounds=config["mvp_rounds"],
    )) as target:
        with target.db.transaction(), target.db.cursor() as cursor:
            target.lock_writes()
            for table, values in rows.items():
                destination = sql.Identifier(f"routing_{table}")
                cursor.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(destination))
                if cursor.fetchone() is not None:
                    raise ValueError("The PostgreSQL destination already contains routing data.")
                if values:
                    placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(values[0]))
                    cursor.executemany(sql.SQL("INSERT INTO {} VALUES ({})").format(destination, placeholders), values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the old SQLite database file.")
    args = parser.parse_args()
    dsn = os.environ.get("ROUTING_DATABASE_URL", "").strip()
    if not dsn:
        parser.error("Set ROUTING_DATABASE_URL to the PostgreSQL destination.")
    try:
        migrate(args.source, dsn)
    except psycopg.Error as exc:
        parser.error(f"PostgreSQL import failed ({type(exc).__name__}). Check the destination and permissions.")
    except (ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        parser.error(f"Could not import routing experience: {exc}")
    print("Routing experience imported into PostgreSQL.")


if __name__ == "__main__":
    main()
