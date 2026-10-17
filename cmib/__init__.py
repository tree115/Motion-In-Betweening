"""Expose the original CMIB import name while retaining this fork's mib layout."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "mib")]
