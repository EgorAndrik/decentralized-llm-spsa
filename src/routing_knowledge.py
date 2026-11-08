"""Persistent local routing experience and the shared snapshot from the last round."""

import json
import sqlite3
from pathlib import Path


class RoutingKnowledge:
    def __init__(
        self,
        path: Path,
        scope: dict,
        *,
        round_size: int | None = None,
        mvp_rounds: int | None = None,
    ):
        if (round_size is not None and round_size < 1) or (mvp_rounds is not None and mvp_rounds < 1):
            raise ValueError("Round size and MVP rounds must be positive.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            with self.db:
                self.db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS experience (
                        controller_id INTEGER NOT NULL,
                        task_type TEXT NOT NULL,
                        agent_id INTEGER NOT NULL,
                        observations INTEGER NOT NULL,
                        successes INTEGER NOT NULL,
                        quality_sum REAL NOT NULL,
                        service_seconds REAL NOT NULL,
                        PRIMARY KEY (controller_id, task_type, agent_id)
                    )
                """)
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS execution_failures (
                        controller_id INTEGER NOT NULL,
                        task_type TEXT NOT NULL,
                        agent_id INTEGER NOT NULL,
                        failures INTEGER NOT NULL,
                        PRIMARY KEY (controller_id, task_type, agent_id)
                    )
                """)
                self.db.execute("CREATE TABLE IF NOT EXISTS evaluated_prompts (digest TEXT PRIMARY KEY)")
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS published_readiness (id INTEGER PRIMARY KEY, ready INTEGER NOT NULL)
                """)
                defaults = {
                    "version": 1, "scope": scope,
                    "round_size": round_size if round_size is not None else 5,
                    "mvp_rounds": mvp_rounds if mvp_rounds is not None else 3,
                }
                self.db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (json.dumps(defaults),))
                config = json.loads(self.db.execute("SELECT value FROM config WHERE id = 1").fetchone()[0])
                if config["version"] != 1 or config["scope"] != scope:
                    raise ValueError("Knowledge database was created for a different mode or model pool.")
                for field, requested in (("round_size", round_size), ("mvp_rounds", mvp_rounds)):
                    if requested is not None and requested != config[field]:
                        raise ValueError(f"Knowledge database already uses {field}={config[field]}.")
                self.round_size = config["round_size"]
                self.mvp_rounds = config["mvp_rounds"]
                self.required_agents = set(range(len(scope["agents"])))
                self.required_controllers = set(range(scope["controllers"]))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.db.close()

    @property
    def observations(self) -> int:
        return self.db.execute(
            "SELECT COALESCE(SUM(observations), 0) FROM experience WHERE controller_id >= 0"
        ).fetchone()[0]

    @property
    def completed_rounds(self) -> int:
        return self.observations // self.round_size

    @property
    def ready(self) -> bool:
        row = self.db.execute("SELECT ready FROM published_readiness WHERE id = 1").fetchone()
        return bool(row and row[0])

    def coverage(self) -> tuple[set[int], set[int], int]:
        pairs = self.db.execute("""
            SELECT DISTINCT agent_id, controller_id FROM experience
            WHERE controller_id >= 0 AND task_type != 'WEB_SEARCH'
        """).fetchall()
        prompts = self.db.execute("SELECT COUNT(*) FROM evaluated_prompts").fetchone()[0]
        return {pair[0] for pair in pairs}, {pair[1] for pair in pairs}, prompts

    @property
    def failures(self) -> int:
        return self.db.execute(
            "SELECT COALESCE(SUM(failures), 0) FROM execution_failures WHERE controller_id >= 0"
        ).fetchone()[0]

    def record_failure(self, controller_id, task_type, agent_id) -> None:
        with self.db:
            self.db.execute("""
                INSERT INTO execution_failures VALUES (?, ?, ?, 1)
                ON CONFLICT (controller_id, task_type, agent_id) DO UPDATE SET failures = failures + 1
            """, (controller_id, task_type, agent_id))

    def record(
        self, controller_id, task_type, agent_id, *, success, quality, service_seconds, prompt_digest=None,
    ) -> int | None:
        """Save one evaluated outcome. Return the round number when a snapshot is published."""
        with self.db:
            self.db.execute("""
                INSERT INTO experience VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT (controller_id, task_type, agent_id) DO UPDATE SET
                    observations = observations + 1,
                    successes = successes + excluded.successes,
                    quality_sum = quality_sum + excluded.quality_sum,
                    service_seconds = service_seconds + excluded.service_seconds
            """, (controller_id, task_type, agent_id, int(success), quality, service_seconds))
            if prompt_digest and task_type != "WEB_SEARCH":
                self.db.execute("INSERT OR IGNORE INTO evaluated_prompts VALUES (?)", (prompt_digest,))
            count = self.observations
            if count % self.round_size:
                return None
            # -1 is the shared snapshot. Rebuild from cumulative local counts,
            # so observations that were published before are only counted once.
            self.db.execute("DELETE FROM experience WHERE controller_id = -1")
            self.db.execute("""
                INSERT INTO experience
                SELECT -1, task_type, agent_id, SUM(observations), SUM(successes),
                       SUM(quality_sum), SUM(service_seconds)
                FROM experience WHERE controller_id >= 0 GROUP BY task_type, agent_id
            """)
            self.db.execute("DELETE FROM execution_failures WHERE controller_id = -1")
            self.db.execute("""
                INSERT INTO execution_failures
                SELECT -1, task_type, agent_id, SUM(failures)
                FROM execution_failures WHERE controller_id >= 0 GROUP BY task_type, agent_id
            """)
            round_number = count // self.round_size
            agents, controllers, prompts = self.coverage()
            ready = (
                round_number >= self.mvp_rounds
                and self.required_agents <= agents
                and self.required_controllers <= controllers
                and prompts >= 2
            )
            self.db.execute("INSERT OR REPLACE INTO published_readiness VALUES (1, ?)", (int(ready),))
            return round_number

    def estimate(self, controller_id, task_type, agent_id, quality_prior, time_prior):
        """Use the controller's experience, or the published snapshot for unseen pairs."""
        row = self.db.execute("""
            SELECT controller_id, observations, successes, quality_sum, service_seconds
            FROM experience WHERE controller_id IN (?, -1) AND task_type = ? AND agent_id = ?
            ORDER BY controller_id DESC LIMIT 1
        """, (controller_id, task_type, agent_id)).fetchone()
        count, successes, quality_sum, service_seconds = row[1:] if row else (0, 0, 0.0, 0.0)
        local_failure = self.db.execute("""
            SELECT failures FROM execution_failures WHERE controller_id = ? AND task_type = ? AND agent_id = ?
        """, (controller_id, task_type, agent_id)).fetchone()
        if local_failure is not None:
            failures = local_failure[0]
            reliability_count = count if row and row[0] == controller_id else 0
        elif row and row[0] == controller_id:
            failures, reliability_count = 0, count
        else:
            shared_failure = self.db.execute("""
                SELECT failures FROM execution_failures WHERE controller_id = -1 AND task_type = ? AND agent_id = ?
            """, (task_type, agent_id)).fetchone()
            failures, reliability_count = (shared_failure[0] if shared_failure else 0), count
        # Three observations' worth of prior dampens the effect of early samples.
        reliability = (reliability_count + 3) / (reliability_count + 3 + failures)
        acceptance = (successes + 3) / (count + 3)
        quality = (quality_sum + 3 * quality_prior) / (count + 3)
        return quality * acceptance * reliability, (service_seconds + 3 * time_prior) / (count + 3)
