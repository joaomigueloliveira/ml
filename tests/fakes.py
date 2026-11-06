from collections.abc import Mapping, Sequence
from typing import Any


class FakeTracker:
    """Records what a run would send to W&B."""

    def __init__(self) -> None:
        self.configured: dict[str, Any] = {}
        self.history: list[dict[str, float]] = []
        self.summarised: dict[str, Any] = {}
        self.tables: dict[str, tuple[list[str], list[list[Any]]]] = {}
        self.finished = False

    def config(self, values: Mapping[str, Any]) -> None:
        self.configured |= values

    def log(self, metrics: Mapping[str, float]) -> None:
        self.history.append(dict(metrics))

    def summary(self, values: Mapping[str, Any]) -> None:
        self.summarised |= values

    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.tables[key] = (list(columns), [list(r) for r in rows])

    def finish(self) -> None:
        self.finished = True
