"""Where the competition data and Callproof's own caches live."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "CALLPROOF_DATA"
CACHE_ENV = "CALLPROOF_CACHE"
KAGGLEHUB_DEFAULT = Path("~/.cache/kagglehub/competitions/gemma-4-developer-agent")
CACHE_DEFAULT = Path("~/.cache/callproof")


def resolve_data(flag: Path | str | None = None) -> Path:
    """`--data` flag, then `CALLPROOF_DATA`, then the directory `kagglehub.competition_download` fills."""
    chosen = flag or os.environ.get(DATA_ENV) or KAGGLEHUB_DEFAULT
    return Path(chosen).expanduser()


def resolve_cache(flag: Path | str | None = None) -> Path:
    """`CALLPROOF_CACHE`, then `~/.cache/callproof`."""
    chosen = flag or os.environ.get(CACHE_ENV) or CACHE_DEFAULT
    return Path(chosen).expanduser()
