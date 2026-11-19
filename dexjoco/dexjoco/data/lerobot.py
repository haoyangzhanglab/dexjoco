"""Read the official v1.1 low-dimensional demonstrations without LeRobot."""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from numpy.typing import NDArray

DATASET_ID = "DexJoCo/DexJoCo-Datasets-v1.1"
DATASET_REVISION = "077bbed4a3c9807ae88dad06842b7989533ab2bc"


def read_episodes(
    dataset_dir: str | Path, episode_indices: Iterable[int] | None = None,
) -> Iterator[tuple[int, NDArray[np.float64], dict]]:
    root = Path(dataset_dir)
    metadata = json.loads((root / "meta/info.json").read_text())
    with (root / "replay_initial_values.jsonl").open() as stream:
        entries = sorted((json.loads(line) for line in stream), key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in entries] != list(range(metadata["total_episodes"])):
        raise ValueError("Initial scenes must cover every episode exactly once")
    indices = list(range(len(entries))) if episode_indices is None else list(episode_indices)
    for index in indices:
        if not 0 <= index < len(entries):
            raise ValueError(f"Episode index {index} is outside [0, {len(entries)})")
    episode_metadata = ds.dataset(root / "meta/episodes", format="parquet").to_table(
        columns=["episode_index", "length"],
    ).to_pydict()
    lengths = dict(zip(episode_metadata["episode_index"], episode_metadata["length"]))
    table = ds.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=ds.field("episode_index").isin(indices),
    )
    for index in indices:
        rows = table.filter(ds.field("episode_index") == index).sort_by("frame_index")
        frames = np.asarray(rows["frame_index"])
        length = lengths[index]
        if not len(frames) or not np.array_equal(frames, np.arange(length)):
            raise ValueError(
                f"Episode {index} requires {length} consecutive frames from index 0, got {len(frames)} rows"
            )
        action_column = rows["action"].combine_chunks()
        widths = np.asarray(action_column.value_lengths())
        if not np.all(widths == widths[0]) or widths[0] == 0:
            raise ValueError(f"Episode {index} has inconsistent action widths")
        actions = np.asarray(action_column.flatten()).reshape(len(rows), int(widths[0]))
        if actions.dtype != np.float64:
            raise ValueError("Replay requires the original v1.1 float64 quaternion actions")
        if not np.isfinite(actions).all():
            raise ValueError(f"Episode {index} has non-finite actions")
        yield index, actions, entries[index]["replay_initial_values"]


def main():
    parser = argparse.ArgumentParser(description="Download low-dimensional v1.1 rand_obj demonstrations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    patterns = [
        f"rand_obj/{task}/{suffix}"
        for task in args.tasks
        for suffix in ("data/**", "meta/episodes/**", "meta/info.json", "replay_initial_values.jsonl")
    ]
    snapshot_download(
        DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=patterns, local_dir=args.output,
    )


if __name__ == "__main__":
    main()
