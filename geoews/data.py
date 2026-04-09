"""Compatibility module re-exporting dataset loaders."""

from __future__ import annotations

from .datasets import load_ngrip, load_peter_lake

__all__ = ["load_ngrip", "load_peter_lake"]

