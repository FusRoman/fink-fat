#!/usr/bin/env python3
from __future__ import annotations

import re
import hashlib
from glob import glob
from pathlib import Path
from typing import Dict, Tuple

import pyarrow.parquet as pq

from config import PARQUET_DIR


# edge_features_001149_left03202_right03207_gap5.parquet
FNAME_RE = re.compile(
    r".*?_left(?P<left>\d+)_right(?P<right>\d+)_gap(?P<gap>\d+)\.parquet$"
)


def parse_group_from_path(path: str) -> Tuple[int, int, int]:
    """
    Parse (left_nid, right_nid, gap_nights) from a parquet filename.

    Parameters
    ----------
    path : str
        File path following the pattern
        `..._leftXXXXX_rightYYYYY_gapZ.parquet`.

    Returns
    -------
    (int, int, int)
        (left_nid, right_nid, gap_nights)

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.
    """
    m = FNAME_RE.match(path)
    if m is None:
        raise ValueError(f"Cannot parse group keys from filename: {path}")
    left = int(m.group("left"))
    right = int(m.group("right"))
    gap = int(m.group("gap"))
    return left, right, gap


def stable_u64(key: str, *, seed: str) -> int:
    """
    Deterministic 64-bit hash for stable splits.

    Parameters
    ----------
    key : str
        Key to hash.
    seed : str
        Salt/seed for the hash, change it only if you want a new split.

    Returns
    -------
    int
        Unsigned 64-bit integer.
    """
    h = hashlib.blake2b((seed + "|" + key).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def choose_split_for_left_night(
    left_nid: int,
    gap: int,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: str,
) -> str:
    """
    Assign a LEFT night to a split deterministically (night-level split).

    Notes
    -----
    - The split decision is *primarily* driven by `left_nid`, so the same night
      never appears in multiple splits (strict anti-leakage).
    - We include `gap` in the hash key to reduce pathological distributions
      where certain gaps would concentrate in a single split.

    Returns
    -------
    str
        One of {"train", "val", "test"}.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    # Night-level key (+gap to keep gap mix healthy)
    key = f"left={left_nid}|gap={gap}"
    x = stable_u64(key, seed=seed) / float(2**64)  # in [0, 1)

    if x < train_frac:
        return "train"
    if x < train_frac + val_frac:
        return "val"
    return "test"


def get_writer(
    writers: Dict[str, pq.ParquetWriter],
    split: str,
    out_dir: Path,
    schema,
) -> pq.ParquetWriter:
    """
    Lazily create a ParquetWriter for the given split.
    """
    if split in writers:
        return writers[split]

    out_path = out_dir / f"{split}.parquet"
    writer = pq.ParquetWriter(
        out_path.as_posix(),
        schema,
        compression="zstd",
        use_dictionary=True,
    )
    writers[split] = writer
    return writer


def main() -> None:
    """
    Merge many edge-feature parquets into train/val/test parquets (night-level split).

    Strategy
    --------
    - Split assignment is done per *left night id* (strict anti-leakage):
      the same `left_nid` never appears in multiple splits.
    - A small "gap-aware" hashing is used to keep each split representative
      across gap_nights as much as possible.
    - Streaming write via iter_batches avoids high memory usage.
    """
    files = sorted(glob(str(PARQUET_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {PARQUET_DIR}")

    out_dir = Path(PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fractions
    train_frac = 0.80
    val_frac = 0.10
    test_frac = 0.10

    # Change seed to deterministically reshuffle the night-level split
    seed = "fink-fat-night-split-v1"

    batch_size = 1_000_000

    writers: Dict[str, pq.ParquetWriter] = {}
    base_schema = None

    file_counts = {"train": 0, "val": 0, "test": 0}
    row_counts = {"train": 0, "val": 0, "test": 0}

    # Useful to verify the strictness
    left_nights_seen: Dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}

    for path in files:
        left, right, gap = parse_group_from_path(path)

        split = choose_split_for_left_night(
            left,
            gap,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )

        print(f"[{split:5s}] {Path(path).name}  (left={left}, right={right}, gap={gap})")

        pf = pq.ParquetFile(path)

        for batch in pf.iter_batches(batch_size=batch_size):
            if base_schema is None:
                base_schema = batch.schema

            if batch.schema != base_schema:
                raise ValueError(
                    "Schema mismatch detected.\n"
                    f"First schema: {base_schema}\n"
                    f"This file : {path}\n"
                    f"Batch schema: {batch.schema}\n"
                    "Tip: ensure all parquet files are produced with the same feature set."
                )

            writer = get_writer(writers, split, out_dir, base_schema)
            writer.write_batch(batch)
            row_counts[split] += batch.num_rows

        file_counts[split] += 1
        left_nights_seen[split].add(left)

    for w in writers.values():
        w.close()

    # Sanity: strict anti-leakage on left nights
    inter_tv = left_nights_seen["train"].intersection(left_nights_seen["val"])
    inter_tt = left_nights_seen["train"].intersection(left_nights_seen["test"])
    inter_vt = left_nights_seen["val"].intersection(left_nights_seen["test"])
    if inter_tv or inter_tt or inter_vt:
        raise RuntimeError(
            "Night-level split violation detected (a left_nid appears in multiple splits).\n"
            f"train∩val: {sorted(inter_tv)[:10]}\n"
            f"train∩test: {sorted(inter_tt)[:10]}\n"
            f"val∩test: {sorted(inter_vt)[:10]}"
        )

    print("\n=== Summary ===")
    for s in ("train", "val", "test"):
        print(
            f"{s:5s}: files={file_counts[s]:4d}  rows={row_counts[s]:12d}  "
            f"unique_left_nights={len(left_nights_seen[s]):4d}"
        )
    print(f"Outputs written to: {out_dir}")
    print("Strict check: OK (no left_nid overlaps across splits)")


if __name__ == "__main__":
    main()
