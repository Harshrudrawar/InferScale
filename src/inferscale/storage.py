import json
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, select, update


class Store:
    """Terminal experiment records are immutable through the application API."""

    def __init__(self, url):
        self.engine = create_engine(
            url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
        )
        meta = MetaData()
        self.table = Table(
            "experiments",
            meta,
            Column("id", String(36), primary_key=True),
            Column("status", String(20), nullable=False),
            Column("created_at", String(40), nullable=False),
            Column("document", Text, nullable=False),
        )
        meta.create_all(self.engine)

    def create(self, document):
        record = {
            **document,
            "id": str(uuid4()),
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self.engine.begin() as conn:
            conn.execute(
                self.table.insert().values(
                    id=record["id"],
                    status="running",
                    created_at=record["created_at"],
                    document=json.dumps(record),
                )
            )
        return record

    def finish(self, id, status, payload):
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Invalid terminal status")
        record = self.get(id)
        if record is None:
            raise KeyError(id)
        record.update(payload, status=status, finished_at=datetime.now(UTC).isoformat())
        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.table)
                .where(self.table.c.id == id, self.table.c.status == "running")
                .values(status=status, document=json.dumps(record))
            )
            if result.rowcount != 1:
                raise ValueError("Experiment is already terminal")
        return record

    def get(self, id):
        with self.engine.connect() as conn:
            doc = conn.execute(
                select(self.table.c.document).where(self.table.c.id == id)
            ).scalar_one_or_none()
        return json.loads(doc) if doc else None

    def list(self, limit=100):
        with self.engine.connect() as conn:
            docs = (
                conn.execute(
                    select(self.table.c.document)
                    .order_by(self.table.c.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        return [json.loads(d) for d in docs]

    def recover(self):
        with self.engine.connect() as conn:
            ids = (
                conn.execute(select(self.table.c.id).where(self.table.c.status == "running"))
                .scalars()
                .all()
            )
        for id in ids:
            self.finish(
                id, "interrupted", {"error": "Process stopped before completion; rerun explicitly"}
            )

    def close(self):
        self.engine.dispose()
