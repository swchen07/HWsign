from __future__ import annotations

from datasets.interfaces.hansig import parse_hansig_filename, split_records


def test_parse_hansig_filename() -> None:
    record = parse_hansig_filename("original_w1_2_3.jpg")
    assert record.label == "genuine"
    assert record.writer_id == "w001"
    assert record.name_id == 2
    assert record.signer_id == "w001_n002"
    assert record.sample_id == 3

    forgery = parse_hansig_filename("forgery_w12_4_9.jpg")
    assert forgery.label == "forgery"
    assert forgery.writer_id == "w012"


def test_split_records_by_writer() -> None:
    records = [
        parse_hansig_filename(f"original_w{writer}_1_{sample}.jpg")
        for writer in range(1, 7)
        for sample in range(1, 3)
    ]
    splits = split_records(records, train_ratio=0.5, val_ratio=0.25, seed=1)
    seen = {}
    for split, items in splits.items():
        for item in items:
            assert item.split == split
            seen.setdefault(item.writer_id, split)
            assert seen[item.writer_id] == split
