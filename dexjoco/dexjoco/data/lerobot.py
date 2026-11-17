"""Read the official v1.1 low-dimensional demonstrations without LeRobot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

DATASET_ID = "DexJoCo/DexJoCo-Datasets-v1.1"
DATASET_REVISION = "077bbed4a3c9807ae88dad06842b7989533ab2bc"


def read_episodes(dataset_dir, episode_indices=None):
    root = Path(dataset_dir)
    metadata = json.loads((root / "meta/info.json").read_text())
    with (root / "replay_initial_values.jsonl").open() as stream:
        entries = sorted((json.loads(line) for line in stream), key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in entries] != list(range(metadata["total_episodes"])):
        raise ValueError("Initial scenes must cover every episode exactly once")
    indices = list(range(len(entries))) if episode_indices is None else list(episode_indices)
    table = ds.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=ds.field("episode_index").isin(indices),
    )
    for index in indices:
        rows = table.filter(ds.field("episode_index") == index).sort_by("frame_index")
        frames = np.asarray(rows["frame_index"])
        if not len(frames) or not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Episode {index} has missing or duplicate frame indices")
        actions = np.asarray(rows["action"].combine_chunks().flatten()).reshape(len(rows), -1)
        if actions.dtype != np.float64:
            raise ValueError("Replay requires the original v1.1 float64 quaternion actions")
        yield index, actions, entries[index]["replay_initial_values"]


def main():
    parser = argparse.ArgumentParser(description="Download v1.1 rand_obj actions and initial scenes; omit videos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    patterns = [
        f"rand_obj/{task}/{suffix}"
        for task in args.tasks
        for suffix in ("data/**", "meta/**", "replay_initial_values.jsonl")
    ]
    snapshot_download(
        DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=patterns, local_dir=args.output,
    )


if __name__ == "__main__":
    main()
