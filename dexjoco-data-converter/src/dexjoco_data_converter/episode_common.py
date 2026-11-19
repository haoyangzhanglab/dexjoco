from pathlib import Path

import numpy as np


def find_episodes(dataset_dir: Path, excluded: list[str] | None) -> list[Path]:
    """List completed raw episodes in name order."""
    episodes = sorted(
        path.parent for path in dataset_dir.glob("*/replay.zarr") if path.is_dir()
    )
    if excluded is not None:
        missing = set(excluded).difference(path.name for path in episodes)
        if missing:
            raise ValueError(f"{dataset_dir}: unknown excluded episodes {sorted(missing)}")
        episodes = [path for path in episodes if path.name not in excluded]
    if not episodes:
        raise ValueError(f"No episodes found in {dataset_dir}")
    return episodes


def find_first_non_static_frame(action: np.ndarray) -> int:
    """Find the first frame where an episode starts changing.

    Args:
        action: Per-step action array ordered by time.

    Returns:
        Index of the first action whose value differs from the following action.

    Raises:
        ValueError: If all adjacent actions are identical.
    """
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("All actions in episode are identical (entirely static)")
