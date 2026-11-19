"""Regenerate synchronized RGB, point clouds and proprioception from v1.1 actions."""

import argparse
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr
from numcodecs import Blosc

from dexjoco.actions import quaternion_to_rotvec
from dexjoco.data.lerobot import DATASET_ID, DATASET_REVISION, read_episodes
from dexjoco.policy_env import DexJoCoEnv

SIMULATOR_REVISION = "12ea8ffea2671072f4630a0c83d9233f308a59d4"


def convert_dataset(
    task: str, dataset_dir: str | Path, output: str | Path, *, episodes: int | None = None,
) -> None:
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bimanual = task.startswith("bimanual_")
    indices = None if episodes is None else range(episodes)

    # Keep the temporary store on the destination filesystem for atomic publication.
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as directory:
        staging = Path(directory) / "dataset.zarr"
        group = zarr.open_group(str(staging), mode="w-")
        fields = group.create_group("data")
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
        buffers: dict[str, list[np.ndarray]] = {
            key: [] for key in ("proprio", "rgb", "point_cloud", "action_rotvec")
        }
        episode_ends = []

        def flush() -> None:
            if not buffers["proprio"]:
                return
            for key, values in buffers.items():
                batch = np.stack(values)
                if key in fields:
                    fields[key].append(batch)
                else:
                    chunk_frames = 1 if key in ("rgb", "point_cloud") else 256
                    fields.create_dataset(
                        key, data=batch,
                        chunks=(chunk_frames, *batch.shape[1:]), compressor=compressor,
                    )
                values.clear()

        with closing(DexJoCoEnv(task, action_format="quaternion")) as env:
            group.attrs.update(
                task_name=task, action_space="eef_rotvec", dt=env.dt,
                dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION,
                simulator_revision=SIMULATOR_REVISION,
                observation_timing="before_action", camera=env.camera,
                offsamples=int(env.env.model.vis.quality.offsamples),
                randomize=env.env.randomize, randomize_dynamics=env.env.randomize_dynamics,
                data_processing=env.data_processing,
            )
            expected_shape = env.action_space.shape
            for index, actions, initial in read_episodes(dataset_dir, indices):
                if actions.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Episode {index} has action shape {actions.shape[1:]}, "
                        f"expected {expected_shape} for {task}"
                    )
                targets = quaternion_to_rotvec(actions, bimanual=bimanual).astype(np.float32)
                observation, _ = env.reset(seed=index, options={"initial_state": initial})
                # Match the upstream replay's final-action extension and termination.
                for step in range(len(actions) + 60):
                    frame = min(step, len(actions) - 1)
                    for key in observation:
                        buffers[key].append(observation[key])
                    buffers["action_rotvec"].append(targets[frame])
                    if len(buffers["proprio"]) == 64:
                        flush()
                    observation, _, terminated, truncated, info = env.step(actions[frame])
                    if info["success"] or terminated or truncated:
                        break
                if not info["success"]:
                    raise RuntimeError(f"{task} episode {index} failed replay")
                flush()
                episode_ends.append(fields["proprio"].shape[0])
                print(f"{task} episode={index} steps={env.action_cnt}", flush=True)
        if not episode_ends:
            raise ValueError("No episodes selected")
        group.create_group("meta").create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
        zarr.consolidate_metadata(group.store)
        staging.rename(output)
    print(f"Saved {len(episode_ends)} episodes, {episode_ends[-1]} frames to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, help="First N episodes; default: all")
    args = parser.parse_args()
    convert_dataset(args.task, args.dataset_dir, args.output, episodes=args.episodes)


if __name__ == "__main__":
    main()
