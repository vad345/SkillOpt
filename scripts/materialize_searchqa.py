#!/usr/bin/env python
"""Materialize a runnable SearchQA split from the shipped ID manifest.

The repo ships only a *split manifest* at ``data/searchqa_id_split/`` — a list of
example **IDs** per split (train/val/test), with no questions, contexts, or
answers. This script joins those IDs against the upstream Hugging Face dataset
(``lucadiliello/searchqa``, matched on its ``key`` field) and writes runnable
items to ``data/searchqa_split/{train,val,test}/items.json`` in the schema the
SearchQA loader expects::

    {"id": ..., "question": ..., "context": ..., "answers": [...]}

Usage::

    uv pip install -e ".[data]"   # one-time: pulls in `datasets`
    python scripts/materialize_searchqa.py
    python scripts/materialize_searchqa.py --manifest data/searchqa_id_split --out data/searchqa_split

After it finishes, train/eval with ``--split_dir data/searchqa_split`` (which is
already the config default, so you can omit the flag for SearchQA).
"""
from __future__ import annotations

import argparse
import json
import os

# Physical split subdirectories the SkillOpt loader reads (SPLIT_NAMES).
SPLITS = ("train", "val", "test")


def load_manifest_ids(manifest_dir: str, split: str) -> list[str]:
    path = os.path.join(manifest_dir, split, "items.json")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [str(it["id"]) for it in items]


def build_lookup(dataset_name: str, id_field: str) -> tuple[dict, list[str]]:
    """Load every split of the upstream dataset, keyed by its ID field."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    lookup: dict[str, dict] = {}
    columns: list[str] = []
    for split_ds in ds.values():
        columns = split_ds.column_names
        if id_field not in columns:
            raise SystemExit(
                f"Dataset {dataset_name!r} has no {id_field!r} column. "
                f"Available columns: {columns}. Pass --id_field to override."
            )
        for row in split_ds:
            lookup[str(row[id_field])] = row
    return lookup, columns


def to_item(item_id: str, row: dict) -> dict:
    answers = row.get("answers")
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []
    return {
        "id": item_id,
        "question": row.get("question", ""),
        "context": row.get("context", ""),
        "answers": list(answers),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/searchqa_id_split", help="ID manifest directory (input)")
    ap.add_argument("--out", default="data/searchqa_split", help="materialized split directory (output)")
    ap.add_argument("--dataset", default="lucadiliello/searchqa", help="upstream HF dataset name")
    ap.add_argument("--id_field", default="key", help="upstream field matched against manifest IDs")
    args = ap.parse_args()

    print(f"[1/3] Loading upstream dataset {args.dataset!r} (this may download a few hundred MB) ...")
    try:
        lookup, columns = build_lookup(args.dataset, args.id_field)
    except ModuleNotFoundError:
        raise SystemExit("The 'datasets' package is required. Install it with: uv pip install -e \".[data]\"")
    print(f"      loaded {len(lookup)} examples; columns = {columns}")

    print(f"[2/3] Joining manifest IDs from {args.manifest!r} ...")
    total_missing = 0
    for split in SPLITS:
        ids = load_manifest_ids(args.manifest, split)
        items, missing = [], 0
        for item_id in ids:
            row = lookup.get(item_id)
            if row is None:
                missing += 1
                continue
            items.append(to_item(item_id, row))
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        total_missing += missing
        print(f"      {split}: wrote {len(items)}/{len(ids)} items ({missing} IDs not found)")

    print(f"[3/3] Done. Materialized split at {args.out}/")
    if total_missing:
        print(
            f"      WARNING: {total_missing} manifest IDs were not found upstream. "
            f"If this is most of them, the {args.id_field!r} field likely doesn't match — "
            "inspect the printed columns and re-run with --id_field."
        )


if __name__ == "__main__":
    main()
