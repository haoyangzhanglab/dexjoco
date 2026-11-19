#!/usr/bin/env python3
"""Replay recorded quaternion actions and export the resulting camera observations."""

import random
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from absl import app, flags
from tqdm import tqdm

from dexjoco.data.demo_export import write_demo
from dexjoco.data.depth_capture import collect_depth_frames
from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state

FLAGS = flags.FLAGS
flags.DEFINE_enum("exp_name", "water_plant", list(CONFIG_MAPPING), "Task name.")
flags.DEFINE_string(
    "input_dir",
    "./",
    "Directory containing recorded demo folders (each holds a replay.zarr).",
)
flags.DEFINE_string(
    "out_dir",
    "./replay_output",
    "Output base directory for the new zarr and videos.",
)
flags.DEFINE_integer("video_fps", 30, "Playback FPS for saved MP4 videos", lower_bound=1)
flags.DEFINE_bool(
    "randomize",
    True,
    "Enable environment randomization (random preset camera, lighting, texture) at reset",
)
flags.DEFINE_integer(
    "seed",
    0,
    "Base seed for the replay environment; the demo index is added per demo",
    lower_bound=0,
)
flags.DEFINE_bool(
    "save_failed",
    False,
    "Save the replayed demo even when the environment does not report success",
)
flags.DEFINE_bool(
    "restore_state",
    True,
    "Restore the recorded initial scene state (table height + object poses) from state[0]",
)
flags.DEFINE_bool(
    "save_depth",
    False,
    "Also render a depth_array per RGB camera and save <cam>_depth.npz + .mp4 alongside each video",
)
flags.DEFINE_bool(
    "camera_screen_effect",
    False,
    "(bimanual_photograph only) Enable the camera live-screen effect "
    "(live preview + shutter flash + frozen photo). When False, the screen "
    "geom is hidden entirely.",
)
flags.DEFINE_bool(
    "ipad_screen_effect",
    False,
    "(bimanual_unlock_ipad only) Enable the iPad unlock screen fade-in "
    "transition. When False, the screen flips to unlocked instantly on the "
    "final correct press.",
)


def _read_demo(path: Path, *, bimanual: bool) -> tuple[np.ndarray, np.ndarray | None]:
    data = zarr.open_group(str(path), mode="r")["data"]
    actions = np.asarray(data["action"], dtype=np.float64)
    action_dim = 46 if bimanual else 23
    if actions.ndim != 2 or actions.shape[1] != action_dim or not len(actions):
        raise ValueError(f"{path}: expected nonempty (T, {action_dim}) actions, got {actions.shape}.")
    quaternions = actions.reshape(len(actions), -1, 23)[..., 3:7]
    if not np.isfinite(actions).all() or not np.any(quaternions, axis=-1).all():
        raise ValueError(f"{path}: actions must be finite with nonzero quaternions.")

    initial_state = None
    if FLAGS.restore_state:
        state = data["state"]
        if state.ndim != 2 or state.shape[0] != len(actions):
            raise ValueError(f"{path}: expected one state vector per action, got {state.shape}.")
        initial_state = np.asarray(state[0], dtype=np.float64)
    return actions, initial_state


def _replay_demo(input_path: Path, output_dir: Path, index: int) -> tuple[bool, Path | None]:
    task_id = FLAGS.exp_name
    bimanual = task_id.startswith("bimanual_")
    actions, initial_state = _read_demo(input_path, bimanual=bimanual)
    config = CONFIG_MAPPING[task_id]()
    env_kwargs = {}
    if task_id == "bimanual_photograph":
        env_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    elif task_id == "bimanual_unlock_ipad":
        env_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect

    seed = FLAGS.seed + index
    random.seed(seed)
    np.random.seed(seed)
    with closing(config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=FLAGS.randomize,
        randomize_dynamics=False,
        seed=seed,
        **env_kwargs,
    )) as env:
        obs, _ = env.reset()
        if initial_state is not None:
            if initial_state.shape != obs["state"].shape or not np.isfinite(initial_state).all():
                raise ValueError(f"{input_path}: initial state must match the selected task and be finite.")
            obs = restore_initial_state(env, task_id, config, initial_state)

        trajectory: list[dict] = []
        success = False
        raw_env = env.unwrapped
        for action in tqdm(actions, desc=input_path.parent.name):
            transition = {"observations": obs, "actions": action}
            if FLAGS.save_depth:
                transition["depth"] = collect_depth_frames(env, obs)
            trajectory.append(transition)

            # The stored right/left blocks match the raw task's action layout.
            command = {"right": action[:23], "left": action[23:]} if bimanual else action
            raw_obs, _, terminated, truncated, info = raw_env.step(command)
            success = bool(info["succeed"])
            if success or terminated or truncated:
                break
            obs = env.observation(raw_obs)
        control_dt = raw_env.control_dt

    print(f"{input_path.parent.name}: success={success}, steps={len(trajectory)}")
    if not success and not FLAGS.save_failed:
        return False, None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = output_dir / f"{task_id}_demo_{index}_replay_{timestamp}"
    write_demo(output, trajectory, control_dt, FLAGS.video_fps)
    return success, output


def main(_argv):
    inputs = sorted(Path(FLAGS.input_dir).glob("*/replay.zarr"))
    if not inputs:
        raise FileNotFoundError(f"No replay.zarr found under {FLAGS.input_dir}.")

    output_dir = Path(FLAGS.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = successful = 0
    for index, input_path in enumerate(inputs, start=1):
        success, output = _replay_demo(input_path, output_dir, index)
        successful += int(success)
        if output is not None:
            print(f"[Saved] {output}")
            saved += 1

    print(f"Successful: {successful}/{len(inputs)}. Saved: {saved}/{len(inputs)}.")


if __name__ == "__main__":
    app.run(main)
