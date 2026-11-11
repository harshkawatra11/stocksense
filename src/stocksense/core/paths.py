from __future__ import annotations

from pathlib import Path

from stocksense.core.config import Settings


class Paths:
    """Every D.4 data folder, derived from Settings.paths.data_root."""

    def __init__(self, settings: Settings) -> None:
        self.root: Path = Path(settings.paths.data_root)

    @property
    def parquet(self) -> Path:
        return self.root / "parquet"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def news(self) -> Path:
        return self.root / "news"

    @property
    def panel(self) -> Path:
        return self.root / "panel"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def preds(self) -> Path:
        return self.root / "preds"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def archive(self) -> Path:
        return self.root / "_archive_v1"

    @property
    def sqlite(self) -> Path:
        return self.state / "stocksense.sqlite"

    @property
    def heartbeat(self) -> Path:
        return self.state / "heartbeat.json"

    def dataset_dir(self, name: str) -> Path:
        """Folder holding the parquet partitions of a raw dataset."""
        return self.parquet / name
