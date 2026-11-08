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
        return self.completed_rounds >= self.mvp_rounds

    def record(self, controller_id, task_type, agent_id, *, success, quality, service_seconds) -> int | None:
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
            return count // self.round_size

    def estimate(self, controller_id, task_type, agent_id, quality_prior, time_prior):
        """Use the controller's experience, or the published snapshot for unseen pairs."""
        row = self.db.execute("""
            SELECT observations, quality_sum, service_seconds
            FROM experience WHERE controller_id IN (?, -1) AND task_type = ? AND agent_id = ?
            ORDER BY controller_id DESC LIMIT 1
        """, (controller_id, task_type, agent_id)).fetchone()
        if row is None:
            return quality_prior, time_prior
        count, quality_sum, service_seconds = row
        # Three observations' worth of prior dampens the effect of early samples.
        return (quality_sum + 3 * quality_prior) / (count + 3), (service_seconds + 3 * time_prior) / (count + 3)
