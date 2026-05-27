from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignatureRecord:
    image_path: str
    label: str
    writer_id: str
    name_id: int
    signer_id: str
    sample_id: int
    source_dataset: str = "hansig"
    split: str | None = None

    @property
    def is_genuine(self) -> bool:
        return self.label == "genuine"

    def with_split(self, split: str) -> SignatureRecord:
        return SignatureRecord(
            image_path=self.image_path,
            label=self.label,
            writer_id=self.writer_id,
            name_id=self.name_id,
            signer_id=self.signer_id,
            sample_id=self.sample_id,
            source_dataset=self.source_dataset,
            split=split,
        )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        if row["split"] is None:
            row["split"] = ""
        return row


def ensure_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def records_to_rows(records: Iterable[SignatureRecord]) -> list[dict[str, object]]:
    return [record.to_row() for record in records]
