#!/usr/bin/env python
"""Materialize a runnable LiveMathematicianBench split from the shipped ID manifest.

``data/livemathematicianbench_id_split/`` ships only IDs (``<month>:<no>``) per
split, plus the upstream ``source_file`` each ID came from. This script:

  1. downloads the upstream monthly ``qa_*_final.json`` files from the HF dataset
     repo recorded in ``split_manifest.json`` (pinned to its revision), and
  2. normalizes each raw item with the project's own loader
     (``_normalize_item``) so the schema is guaranteed to match the rollout, then
  3. selects the manifest IDs and writes runnable items to
     ``data/livemathematicianbench_split/{train,val,test}/items.json``.

Usage::

    uv pip install -e ".[data]"            # one-time: pulls in datasets + huggingface_hub
    python scripts/materialize_livemath.py
    # offline (already have the monthly files locally):
    python scripts/materialize_livemath.py --raw_dir /path/to/dir/with/qa_*_final.json

After it finishes, train/eval with ``--split_dir data/livemathematicianbench_split``
(which is already the config default, so you can omit the flag).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SPLITS = ("train", "val", "test")

# Make the project importable when run from the repo root.
sys.path.insert(0, os.getcwd())


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def month_from_source_file(rel: str) -> str:
    """Extract the 6-digit month (e.g. '202602') from a source_file path."""
    for part in rel.replace("\\", "/").split("/"):
        if len(part) == 6 and part.isdigit():
            return part
    # Fallback: qa_<month>_final.json
    base = os.path.basename(rel)
    for token in base.replace(".", "_").split("_"):
        if len(token) == 6 and token.isdigit():
            return token
    return ""


def resolve_local_file(rel: str, raw_dir: str) -> str | None:
    """Find the local copy of an upstream source_file inside raw_dir."""
    direct = os.path.join(raw_dir, rel)
    if os.path.isfile(direct):
        return direct
    matches = glob.glob(os.path.join(raw_dir, "**", os.path.basename(rel)), recursive=True)
    return matches[0] if matches else None


def index_monthly_file(local_path: str, month: str) -> dict[str, dict]:
    """Return {str(no): normalized_item} for one monthly qa_*_final.json file."""
    from skillopt.envs.livemathematicianbench.dataloader import _normalize_item

    raw = read_json(local_path)
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array in {local_path}, got {type(raw).__name__}")
    by_no: dict[str, dict] = {}
    for row_idx, item in enumerate(raw):
        merged = dict(item)
        merged.setdefault("month", month)  # raw items carry month implicitly via path
        norm = _normalize_item(merged, row_idx=row_idx, source_path=local_path)
        if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
            by_no[str(norm["no"])] = norm
    return by_no


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/livemathematicianbench_id_split", help="ID manifest directory (input)")
    ap.add_argument("--out", default="data/livemathematicianbench_split", help="materialized split directory (output)")
    ap.add_argument("--raw_dir", default="", help="local dir containing qa_*_final.json (skips HF download)")
    args = ap.parse_args()

    meta = read_json(os.path.join(args.manifest, "split_manifest.json"))
    repo_id = meta["source_repo"]
    revision = meta.get("source_revision")
    source_files = meta["source_files"]

    # 1. Locate each upstream monthly file (local or downloaded), and index by `no`.
    print(f"[1/3] Resolving {len(source_files)} monthly source files "
          f"({'local: ' + args.raw_dir if args.raw_dir else repo_id + '@' + str(revision)[:8]}) ...")
    file_index: dict[str, dict[str, dict]] = {}
    if not args.raw_dir:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError:
            raise SystemExit("huggingface_hub is required. Install it with: uv pip install -e \".[data]\"")
    for rel in source_files:
        month = month_from_source_file(rel)
        if args.raw_dir:
            local = resolve_local_file(rel, args.raw_dir)
            if local is None:
                print(f"      WARNING: {rel} not found under {args.raw_dir}; skipping")
                continue
        else:
            local = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision)
        file_index[rel] = index_monthly_file(local, month)
        print(f"      {rel}: indexed {len(file_index[rel])} items (month={month})")

    # 2. Join manifest IDs against the indexed upstream items.
    print(f"[2/3] Joining manifest IDs from {args.manifest} ...")
    total_missing = 0
    for split in SPLITS:
        manifest_items = read_json(os.path.join(args.manifest, split, "items.json"))
        selected, missing = [], 0
        for m in manifest_items:
            by_no = file_index.get(m["source_file"], {})
            norm = by_no.get(str(m["no"]))
            if norm is None:
                missing += 1
                continue
            # Carry manifest metadata through if the raw item lacked it.
            norm = {**norm, "paper_link": norm.get("paper_link") or m.get("paper_link", "")}
            selected.append(norm)
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        total_missing += missing
        print(f"      {split}: wrote {len(selected)}/{len(manifest_items)} items ({missing} IDs not found)")

    print(f"[3/3] Done. Materialized split at {args.out}/")
    if total_missing:
        print(f"      WARNING: {total_missing} manifest IDs were not resolved upstream. "
              "Check that the source revision matches and that the monthly files were downloaded.")


if __name__ == "__main__":
    main()
