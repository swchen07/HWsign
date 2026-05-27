"""Dataset interface implementations."""

from datasets.interfaces.base import SignatureRecord
from datasets.interfaces.hansig import (
    HANSIG_FILENAME_RE,
    build_hansig_manifest,
    parse_hansig_filename,
    split_records,
)

__all__ = [
    "HANSIG_FILENAME_RE",
    "SignatureRecord",
    "build_hansig_manifest",
    "parse_hansig_filename",
    "split_records",
]
