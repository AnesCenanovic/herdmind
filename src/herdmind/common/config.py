"""YAML -> frozen dataclass loader. Unknown keys and wrong types fail loudly."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


def load_config(path: str | Path, schema: type[T]) -> T:
    """Merge the YAML at ``path`` over ``schema``'s defaults and return a ``schema``."""
    merged = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(path))
    return OmegaConf.to_object(merged)  # type: ignore[return-value]
