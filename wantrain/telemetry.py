from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class JsonlEventWriter:
    def __init__(self, path: Path, *, run_id: str, process_role: str = "reference") -> None:
        self.path = path
        self.run_id = run_id
        self.process_role = process_role
        self.process_id = f"{process_role}-0"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(self, event_type: str, **payload: Any) -> None:
        event = {
            "schema_version": 1,
            "run_id": self.run_id,
            "event_id": uuid4().hex,
            "event_type": event_type,
            "process_role": self.process_role,
            "process_id": self.process_id,
            "worker_id": None,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_time_ns": time.monotonic_ns(),
        }
        event.update(payload)
        self._handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlEventWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

