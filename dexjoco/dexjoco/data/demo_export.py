"""Write one teleoperation episode as Zarr data and synchronized camera videos."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numcodecs
import numpy as np
import zarr

from dexjoco.actions import quaternion_to_rotvec
from dexjoco.data.depth_capture import write_depth_outputs
from dexjoco.data.video_writer import write_video


def write_demo(demo_dir: Path, trajectory: list[dict], control_dt: float, video_fps: int) -> None:
    observations = [step["observations"] for step in trajectory]
    actions = np.stack([step["actions"] for step in trajectory])
    episode_data = {
        "action": actions,
        "action_rotvec": quaternion_to_rotvec(actions, bimanual=actions.shape[1] == 46),
        "state": np.stack([obs["state"] for obs in observations]),
        "timestamp": np.arange(len(actions)) * control_dt,
    }

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    # Both replay and conversion discover episodes by the presence of replay.zarr.
    with TemporaryDirectory(prefix=".demo-", dir=demo_dir.parent) as staging:
        episode_dir = Path(staging) / demo_dir.name
        videos_dir = episode_dir / "videos"
        videos_dir.mkdir(parents=True)
        for key in observations[0]:
            if key != "state":
                write_video(videos_dir / f"{key}.mp4", (obs[key] for obs in observations), video_fps)

        if "depth" in trajectory[0]:
            write_depth_outputs(
                {key: [step["depth"][key] for step in trajectory] for key in trajectory[0]["depth"]},
                videos_dir,
                video_fps,
            )

        root = zarr.open_group(str(episode_dir / "replay.zarr"), mode="w-")
        data = root.create_group("data")
        for name, array in episode_data.items():
            data.create_dataset(name, data=array, compressor=compressor)
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.array([len(actions)], dtype=np.int64), compressor=None,
        )
        episode_dir.rename(demo_dir)
