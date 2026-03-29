from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config.env_config import get_env
from storage.db import PROJECT_ROOT


DEFAULT_AUDIO_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "audio_outbox"


def get_default_audio_artifacts_dir() -> Path:
    configured_path = get_env("JATAYU_AUDIO_SPOOL_DIR")
    if configured_path:
        return Path(configured_path)

    return DEFAULT_AUDIO_ARTIFACTS_DIR


class AudioArtifactStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else get_default_audio_artifacts_dir()

    def create_audio_file(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str | None = None,
        file_name: str | None = None,
        prefix: str = "reply",
    ) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._resolve_suffix(mime_type=mime_type, file_name=file_name)
        safe_prefix = self._sanitize_prefix(prefix)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        unique_name = f"{safe_prefix}-{timestamp}-{uuid4().hex}{suffix}"
        file_path = self.root_dir / unique_name
        file_path.write_bytes(audio_bytes)
        return file_path

    def resolve_managed_path(self, file_path: str | Path) -> Path:
        managed_path = Path(file_path).resolve(strict=False)
        root_path = self.root_dir.resolve(strict=False)
        if managed_path != root_path and root_path not in managed_path.parents:
            raise ValueError(f"Audio file path is outside the managed spool: {file_path}")

        return managed_path

    def delete(self, file_path: str | Path) -> None:
        managed_path = self.resolve_managed_path(file_path)
        managed_path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_suffix(
        *,
        mime_type: str | None,
        file_name: str | None,
    ) -> str:
        if file_name:
            suffix = Path(file_name).suffix
            if suffix:
                return suffix

        if mime_type:
            guessed_suffix = mimetypes.guess_extension(mime_type, strict=False)
            if guessed_suffix:
                return guessed_suffix

        return ".wav"

    @staticmethod
    def _sanitize_prefix(prefix: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", prefix).strip("-")
        return sanitized or "reply"
