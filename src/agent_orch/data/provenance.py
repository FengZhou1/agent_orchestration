from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StaleArtifactError(RuntimeError):
    """An artifact derived from a different input than the one now in use."""


def assert_same_library(payload: dict, library_path: str | Path) -> str:
    """Require ``payload`` to have been derived from the library at ``library_path``.

    Composition references are keyed by *library index*, so once the library is
    rebuilt the index-to-deployment mapping is no longer known to hold.  Scoring or
    distilling from a stale reference silently measures a composition belonging to
    some other deployment and reports it as the solver's own optimum, which is how a
    two-day-old test reference passed every other resume check while pointing at the
    wrong deployments.  Returns the shared digest.
    """

    current = file_sha256(library_path)
    recorded = payload.get("library_json_sha256")
    if recorded is None:
        raise StaleArtifactError(
            f"artifact records no deployment-library hash, so it cannot be shown to "
            f"match {library_path} ({current[:16]}); regenerate it"
        )
    if str(recorded) != current:
        raise StaleArtifactError(
            f"artifact was derived from deployment library {str(recorded)[:16]}, but "
            f"{library_path} is {current[:16]}; regenerate it"
        )
    return current


@dataclass(frozen=True)
class DatasetManifest:
    artifact_type: str
    source_name: str
    source_url: str
    source_version: str
    license: str
    source_checksum_sha256: str
    preprocessing_command: str
    random_seed: int
    split_boundaries: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def validate(self) -> None:
        required = (
            self.artifact_type,
            self.source_name,
            self.source_url,
            self.source_version,
            self.license,
            self.source_checksum_sha256,
            self.preprocessing_command,
        )
        if any(not value for value in required):
            raise ValueError("Dataset manifest fields must be non-empty")
        if len(self.source_checksum_sha256) != 64:
            raise ValueError("source_checksum_sha256 must be a SHA-256 hex digest")

    def write(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "DatasetManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(**raw)
        manifest.validate()
        return manifest
