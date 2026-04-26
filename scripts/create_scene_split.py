"""
Create train/validation splits for parsed DriveLM data at the scene level.

Why scene-level?
    Frames and QA pairs from the same scene are highly correlated. Splitting by
    row would leak scene context from train into validation and inflate results.

Usage:
    python3 create_scene_split.py \
        --input-dir ./drivelm_parsed \
        --output-dir ./drivelm_splits \
        --train-scenes 12 \
        --val-scenes 3 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


TABLES = ["qa_enriched", "frames", "objects", "scenes"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory with parsed CSVs")
    p.add_argument("--output-dir", required=True, help="Directory to write split CSVs")
    p.add_argument("--train-scenes", type=int, default=12)
    p.add_argument("--val-scenes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in TABLES:
        path = input_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        tables[name] = pd.read_csv(path)
    return tables


def choose_scenes(scene_tokens: list[str], train_scenes: int,
                  val_scenes: int, seed: int) -> tuple[list[str], list[str]]:
    total_needed = train_scenes + val_scenes
    if total_needed != len(scene_tokens):
        raise ValueError(
            f"Requested {total_needed} scenes, but found {len(scene_tokens)}. "
            "Adjust --train-scenes/--val-scenes to match the dataset."
        )

    ordered = sorted(scene_tokens)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    val_tokens = sorted(ordered[:val_scenes])
    train_tokens = sorted(ordered[val_scenes:])
    return train_tokens, val_tokens


def filter_by_scenes(df: pd.DataFrame, scene_tokens: list[str]) -> pd.DataFrame:
    return df[df["scene_token"].isin(scene_tokens)].copy()


def save_split_tables(tables: dict[str, pd.DataFrame], output_dir: Path,
                      split_name: str, scene_tokens: list[str]):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        split_df = filter_by_scenes(df, scene_tokens)
        split_df.to_csv(split_dir / f"{name}.csv", index=False)


def summarize_split(name: str, tables: dict[str, pd.DataFrame], scene_tokens: list[str]) -> dict:
    qa_df = filter_by_scenes(tables["qa_enriched"], scene_tokens)
    frames_df = filter_by_scenes(tables["frames"], scene_tokens)
    objects_df = filter_by_scenes(tables["objects"], scene_tokens)

    return {
        "name": name,
        "num_scenes": len(scene_tokens),
        "scene_tokens": scene_tokens,
        "num_frames": int(len(frames_df)),
        "num_objects": int(len(objects_df)),
        "num_qa": int(len(qa_df)),
        "qa_by_category": {
            str(k): int(v)
            for k, v in qa_df["qa_category"].value_counts().sort_index().items()
        },
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tables = load_tables(input_dir)
    scene_tokens = tables["scenes"]["scene_token"].drop_duplicates().tolist()

    train_tokens, val_tokens = choose_scenes(
        scene_tokens=scene_tokens,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_tables(tables, output_dir, "train", train_tokens)
    save_split_tables(tables, output_dir, "val", val_tokens)

    manifest = {
        "seed": args.seed,
        "input_dir": str(input_dir),
        "train": summarize_split("train", tables, train_tokens),
        "val": summarize_split("val", tables, val_tokens),
    }

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Scene split written to:", output_dir)
    print("Train scenes:", len(train_tokens), "| Val scenes:", len(val_tokens))
    print("Val scene tokens:")
    for token in val_tokens:
        print(" ", token)


if __name__ == "__main__":
    main()
