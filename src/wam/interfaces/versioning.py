"""Config + experiment versioning (T-05, FR-10, AC-04).

Contracts:
- ``config_hash`` is deterministic: identical logical content (any key order,
  tuple vs list, pydantic model vs its dump) yields the identical digest.
- Every ``JsonlRunLogger`` record carries ``run_id`` + ``config_hash`` so any
  rollout is traceable to checkpoint + dataset snapshot + config hash (AC-04).
- Config files must declare a top-level ``wam_config_version`` whose major
  matches ``WAM_CONFIG_VERSION`` — otherwise loading fails.
- Torch-free; stdlib + pydantic + yaml only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import yaml
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from wam.interfaces.protocols import INTERFACES_VERSION
from wam.interfaces.schema import SCHEMA_VERSION

WAM_CONFIG_VERSION = "0.1.0"

__all__ = [
    "WAM_CONFIG_VERSION",
    "JsonlRunLogger",
    "RunMetadata",
    "config_hash",
    "load_config",
    "read_git_commit",
]


def _version_major(version: str) -> str:
    return version.split(".", 1)[0]


def _canonicalize(obj: Any) -> Any:
    """Reduce nested config data to JSON-safe primitives with stable semantics.

    pydantic models are dumped in JSON mode; tuples become lists; mapping keys
    become strings. Anything non-representable raises ``TypeError`` — silent
    lossy coercion would break hash traceability.
    """
    if isinstance(obj, BaseModel):
        return _canonicalize(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"config_hash: unsupported type {type(obj).__name__!r}")


def config_hash(obj: Mapping[str, Any] | BaseModel) -> str:
    """SHA-256 hex digest of the canonical JSON form (sorted keys, no whitespace).

    Stable across dict key order and across pydantic-model vs plain-dict input.
    """
    canonical = json.dumps(
        _canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_git_commit(cwd: str | Path | None = None) -> str:
    """Current HEAD commit hash via ``git rev-parse``; ``'unknown'`` if unavailable.

    Never raises: absence of git or of a repository must not block a run,
    but the gap stays visible in the metadata.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunMetadata(BaseModel):
    """Immutable provenance record for one run/rollout (FR-10, AC-04).

    ``checkpoint_ref`` / ``dataset_snapshot_ref`` are ``None`` only for runs
    that genuinely have no model/data (e.g. hardware smoke tests) — evaluation
    gates must reject ``None`` for training/rollout records.

    ``train_episode_ids`` is the ORDERED list of episode ids the run actually
    trained on, in the order they were hashed into ``dataset_snapshot_ref``.
    Order is part of the record, not decoration: the snapshot hash is a
    sequential digest, so a re-sorted list no longer reproduces it. ``None``
    means "this run trained on the complement of a holdout" — every checkpoint
    written before I-8, and the reason the evaluator keeps a second proof path
    rather than requiring a migration of archived checkpoints.

    Adding the field is hash-safe by construction: ``RunMetadata`` is not an
    input to ``config_hash``, so no recorded hash moves.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_hash: str
    git_commit: str
    schema_version: str = SCHEMA_VERSION
    interfaces_version: str = INTERFACES_VERSION
    checkpoint_ref: str | None = None
    dataset_snapshot_ref: str | None = None
    train_episode_ids: tuple[str, ...] | None = None
    created_at: datetime

    @classmethod
    def create(
        cls,
        run_id: str,
        config: Mapping[str, Any] | BaseModel,
        *,
        checkpoint_ref: str | None = None,
        dataset_snapshot_ref: str | None = None,
        train_episode_ids: Sequence[str] | None = None,
        git_commit: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> RunMetadata:
        """Build metadata from a config object; hashes it, reads git, stamps the clock.

        ``clock`` and ``git_commit`` are injectable for deterministic tests.
        """
        return cls(
            run_id=run_id,
            config_hash=config_hash(config),
            git_commit=git_commit if git_commit is not None else read_git_commit(),
            checkpoint_ref=checkpoint_ref,
            dataset_snapshot_ref=dataset_snapshot_ref,
            train_episode_ids=(
                None if train_episode_ids is None else tuple(str(e) for e in train_episode_ids)
            ),
            created_at=clock(),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (datetimes as ISO-8601 strings)."""
        return self.model_dump(mode="json")


class JsonlRunLogger:
    """Append-only JSONL logger; stamps EVERY record with run_id + config_hash.

    Contracts:
    - File is opened in append mode — existing records are never truncated.
    - Stamps win over caller-supplied ``run_id``/``config_hash`` keys.
    - Records must be JSON-serializable mappings; a failed record writes nothing.
    - Usable as a context manager; ``log`` outside an open state raises.
    """

    def __init__(self, path: str | Path, metadata: RunMetadata) -> None:
        self._path = Path(path)
        self._metadata = metadata
        self._file: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    def open(self) -> Self:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def log(self, record: Mapping[str, Any]) -> None:
        """Write one record as a single JSON line, stamped and flushed."""
        if self._file is None:
            raise RuntimeError("JsonlRunLogger.log: logger is not open")
        stamped = dict(record)
        stamped["run_id"] = self._metadata.run_id
        stamped["config_hash"] = self._metadata.config_hash
        line = json.dumps(_canonicalize(stamped), sort_keys=True, separators=(",", ":"))
        self._file.write(line + "\n")
        self._file.flush()

    def log_metadata(self) -> None:
        """Write the full ``RunMetadata`` as a ``kind='run_metadata'`` record."""
        self.log({"kind": "run_metadata", **self._metadata.to_dict()})

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a WAM YAML config; enforce the ``wam_config_version`` gate.

    Raises ``ValueError`` if the document is not a mapping, the key is missing
    or non-string, or its major version differs from ``WAM_CONFIG_VERSION``.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        # ValueError, not TypeError: this is bad file content, not a bad argument.
        raise ValueError(  # noqa: TRY004
            f"{config_path}: top level must be a mapping, got {type(data).__name__}"
        )
    version = data.get("wam_config_version")
    if not isinstance(version, str):
        raise ValueError(  # noqa: TRY004
            f"{config_path}: missing or non-string 'wam_config_version'"
        )
    if _version_major(version) != _version_major(WAM_CONFIG_VERSION):
        raise ValueError(
            f"{config_path}: incompatible wam_config_version {version!r}, "
            f"expected major {_version_major(WAM_CONFIG_VERSION)}.x.x"
        )
    return data
