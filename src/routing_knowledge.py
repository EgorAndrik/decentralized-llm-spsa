"""Persistent local routing experience and the shared snapshot from the last round."""

import json

import psycopg


class RoutingKnowledge:
    def __init__(
        self,
        dsn: str,
        scope: dict,
        *,
        round_size: int | None = None,
        mvp_rounds: int | None = None,
    ):
        if (round_size is not None and round_size < 1) or (mvp_rounds is not None and mvp_rounds < 1):
            raise ValueError("Round size and MVP rounds must be positive.")
        self.db = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        try:
            self.db.isolation_level = psycopg.IsolationLevel.READ_COMMITTED
            with self.db.transaction():
                self.lock_writes()
                self.db.execute("CREATE TABLE IF NOT EXISTS routing_config (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS routing_experience (
                        controller_id INTEGER NOT NULL,
                        task_type TEXT NOT NULL,
                        agent_id INTEGER NOT NULL,
                        observations BIGINT NOT NULL,
                        successes BIGINT NOT NULL,
                        quality_sum DOUBLE PRECISION NOT NULL,
                        service_seconds DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (controller_id, task_type, agent_id)
                    )
                """)
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS routing_execution_failures (
                        controller_id INTEGER NOT NULL,
                        task_type TEXT NOT NULL,
                        agent_id INTEGER NOT NULL,
                        failures BIGINT NOT NULL,
                        PRIMARY KEY (controller_id, task_type, agent_id)
                    )
                """)
                self.db.execute("CREATE TABLE IF NOT EXISTS routing_evaluated_prompts (digest TEXT PRIMARY KEY)")
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS routing_published_readiness (id INTEGER PRIMARY KEY, ready BOOLEAN NOT NULL)
                """)
                defaults = {
                    "version": 1, "scope": scope,
                    "round_size": round_size if round_size is not None else 5,
                    "mvp_rounds": mvp_rounds if mvp_rounds is not None else 3,
                }
                self.db.execute("""
                    INSERT INTO routing_config VALUES (1, %s) ON CONFLICT (id) DO NOTHING
                """, (json.dumps(defaults),))
                config = json.loads(self.db.execute("SELECT value FROM routing_config WHERE id = 1").fetchone()[0])
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

    def lock_writes(self) -> None:
        # Serialize setup and publications, including when two runners hit a round boundary.
        self.db.execute("SELECT pg_advisory_xact_lock(21317, 1)")

    def close(self) -> None:
        self.db.close()

    @property
    def observations(self) -> int:
        return self.db.execute(
            "SELECT COALESCE(SUM(observations), 0)::bigint FROM routing_experience WHERE controller_id >= 0"
        ).fetchone()[0]

    @property
    def completed_rounds(self) -> int:
        return self.observations // self.round_size

    @property
    def ready(self) -> bool:
        row = self.db.execute("SELECT ready FROM routing_published_readiness WHERE id = 1").fetchone()
        return bool(row and row[0])

    def coverage(self) -> tuple[set[int], set[int], int]:
        pairs = self.db.execute("""
            SELECT DISTINCT agent_id, controller_id FROM routing_experience
            WHERE controller_id >= 0 AND task_type != 'WEB_SEARCH'
        """).fetchall()
        prompts = self.db.execute("SELECT COUNT(*) FROM routing_evaluated_prompts").fetchone()[0]
        return {pair[0] for pair in pairs}, {pair[1] for pair in pairs}, prompts

    @property
    def failures(self) -> int:
        return self.db.execute(
            "SELECT COALESCE(SUM(failures), 0)::bigint FROM routing_execution_failures WHERE controller_id >= 0"
        ).fetchone()[0]

    def record_failure(self, controller_id, task_type, agent_id) -> None:
        with self.db.transaction():
            self.lock_writes()
            self.db.execute("""
                INSERT INTO routing_execution_failures VALUES (%s, %s, %s, 1)
                ON CONFLICT (controller_id, task_type, agent_id) DO UPDATE SET
                    failures = routing_execution_failures.failures + 1
            """, (controller_id, task_type, agent_id))

    def record(
        self, controller_id, task_type, agent_id, *, success, quality, service_seconds, prompt_digest=None,
    ) -> int | None:
        """Save one evaluated outcome. Return the round number when a snapshot is published."""
        with self.db.transaction():
            self.lock_writes()
            self.db.execute("""
                INSERT INTO routing_experience VALUES (%s, %s, %s, 1, %s, %s, %s)
                ON CONFLICT (controller_id, task_type, agent_id) DO UPDATE SET
                    observations = routing_experience.observations + 1,
                    successes = routing_experience.successes + excluded.successes,
                    quality_sum = routing_experience.quality_sum + excluded.quality_sum,
                    service_seconds = routing_experience.service_seconds + excluded.service_seconds
            """, (controller_id, task_type, agent_id, int(success), quality, service_seconds))
            if prompt_digest and task_type != "WEB_SEARCH":
                self.db.execute("""
                    INSERT INTO routing_evaluated_prompts VALUES (%s) ON CONFLICT (digest) DO NOTHING
                """, (prompt_digest,))
            count = self.observations
            if count % self.round_size:
                return None
            # -1 is the shared snapshot. Rebuild from cumulative local counts,
            # so observations that were published before are only counted once.
            self.db.execute("DELETE FROM routing_experience WHERE controller_id = -1")
            self.db.execute("""
                INSERT INTO routing_experience
                SELECT -1, task_type, agent_id, SUM(observations), SUM(successes),
                       SUM(quality_sum), SUM(service_seconds)
                FROM routing_experience WHERE controller_id >= 0 GROUP BY task_type, agent_id
            """)
            self.db.execute("DELETE FROM routing_execution_failures WHERE controller_id = -1")
            self.db.execute("""
                INSERT INTO routing_execution_failures
                SELECT -1, task_type, agent_id, SUM(failures)
                FROM routing_execution_failures WHERE controller_id >= 0 GROUP BY task_type, agent_id
            """)
            round_number = count // self.round_size
            agents, controllers, prompts = self.coverage()
            ready = (
                round_number >= self.mvp_rounds
                and self.required_agents <= agents
                and self.required_controllers <= controllers
                and prompts >= 2
            )
            self.db.execute("""
                INSERT INTO routing_published_readiness VALUES (1, %s)
                ON CONFLICT (id) DO UPDATE SET ready = excluded.ready
            """, (ready,))
            return round_number

    def estimate(self, controller_id, task_type, agent_id, quality_prior, time_prior):
        """Use the controller's experience, or the published snapshot for unseen pairs."""
        row = self.db.execute("""
            SELECT controller_id, observations, successes, quality_sum, service_seconds
            FROM routing_experience WHERE controller_id IN (%s, -1) AND task_type = %s AND agent_id = %s
            ORDER BY controller_id DESC LIMIT 1
        """, (controller_id, task_type, agent_id)).fetchone()
        count, successes, quality_sum, service_seconds = row[1:] if row else (0, 0, 0.0, 0.0)
        local_failure = self.db.execute("""
            SELECT failures FROM routing_execution_failures WHERE controller_id = %s AND task_type = %s AND agent_id = %s
        """, (controller_id, task_type, agent_id)).fetchone()
        if local_failure is not None:
            failures = local_failure[0]
            reliability_count = count if row and row[0] == controller_id else 0
        elif row and row[0] == controller_id:
            failures, reliability_count = 0, count
        else:
            shared_failure = self.db.execute("""
                SELECT failures FROM routing_execution_failures WHERE controller_id = -1 AND task_type = %s AND agent_id = %s
            """, (task_type, agent_id)).fetchone()
            failures, reliability_count = (shared_failure[0] if shared_failure else 0), count
        # Three observations' worth of prior dampens the effect of early samples.
        reliability = (reliability_count + 3) / (reliability_count + 3 + failures)
        acceptance = (successes + 3) / (count + 3)
        quality = (quality_sum + 3 * quality_prior) / (count + 3)
        return quality * acceptance * reliability, (service_seconds + 3 * time_prior) / (count + 3)
