"""Evaluate PI 0.5 policies on DexJoCo simulation environments."""

import multiprocessing as mp
import os
import random
import signal
from collections import deque
from contextlib import ExitStack, closing
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Literal

import imageio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .dexjoco_openpi_env import DexJoCoOpenPIEnv


def _set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def _interp_rotvec_geodesic(
    rotvec0: np.ndarray, rotvec1: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate rotation vectors on SO(3) instead of component-wise lerp."""
    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    relative_rotvec = (r0.inv() * r1).as_rotvec()
    return (r0 * R.from_rotvec(relative_rotvec * t)).as_rotvec()


def _interpolate_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float, dual_arm: bool
) -> np.ndarray:
    interp_action = (1.0 - t) * old_action + t * new_action
    for offset in ((0, 22) if dual_arm else (0,)):
        rotation = slice(offset + 3, offset + 6)
        interp_action[rotation] = _interp_rotvec_geodesic(
            old_action[rotation], new_action[rotation], t
        )
    return interp_action


def inference_process(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    port: int,
    seed: int,
    host: str,
):
    from openpi_client import websocket_client_policy

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_seed(seed)
    with closing(websocket_client_policy.WebsocketClientPolicy(host=host, port=port)) as client:
        while (obs := obs_queue.get()) is not None:
            action_queue.put(client.infer(obs)["actions"])


def _take_action_chunk(
    action_queue: mp.Queue,
    worker: BaseProcess,
    action_dim: int,
    *,
    wait: bool = False,
) -> np.ndarray | None:
    while True:
        try:
            chunk = action_queue.get(timeout=0.1) if wait else action_queue.get_nowait()
            break
        except Empty:
            if not worker.is_alive():
                raise RuntimeError(f"OpenPI inference worker exited with code {worker.exitcode}")
            if not wait:
                return None

    actions = np.asarray(chunk)
    if actions.ndim != 2 or actions.shape[1] != action_dim or not len(actions):
        raise ValueError(f"OpenPI must return a nonempty action chunk of shape (T, {action_dim})")
    if actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("OpenPI actions must be real, finite numbers")
    return actions


def _merge_action_chunk(
    actions: np.ndarray,
    actions_buffer: deque[np.ndarray],
    elapsed_steps: int,
    dual_arm: bool,
) -> None:
    """Discard elapsed actions and blend the remaining overlap with the buffer."""
    actions = actions[elapsed_steps:]
    overlap = min(len(actions), len(actions_buffer))
    for index in range(overlap):
        actions_buffer[index] = _interpolate_action(
            actions_buffer[index], actions[index], (index + 1) / (overlap + 1), dual_arm,
        )
    actions_buffer.extend(actions[overlap:])


def _append_video_frames(video_writers: dict, raw_images: dict):
    for cam_name, writer in video_writers.items():
        writer.append_data(raw_images[cam_name])


def main(
    config: Path,
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    port: int = 8000,
    host: str = "0.0.0.0",
    output: Path | None = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    replan_ratio: float = 0.8,
    episodes: int = 50,
    pad_state_dim46: bool = False,
    record_pressed_digits: bool | None = None,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 < replan_ratio <= 1:
        raise ValueError("replan_ratio must be in (0, 1]")
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    # Load evaluation configuration.
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    robot_type = cfg["robot_type"]
    dual_arm = robot_type == "dual_arm"
    prompt = cfg["prompt"]
    action_dim = 44 if dual_arm else 22

    # Record password input only for iPad tasks unless explicitly configured.
    if record_pressed_digits is None:
        record_pressed_digits = env_name == "bimanual_unlock_ipad"

    # Write episode videos under a temporary name before assigning the result suffix.
    if output is None:
        output_dir = (
            Path("outputs")
            / f"{env_name}{'_rand_full' if rand_full else ''}_seed{seed}"
        )
    else:
        output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the DexJoCo environment wrapper used by the OpenPI policy.
    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=randomize_dynamics,
        dual_arm=dual_arm,
        prompt=prompt,
        render_mode=render_mode,
        pad_state_dim46=pad_state_dim46,
        password=cfg.get("password", None),  # Pass password from config if available
    )
    # Start the worker without inheriting the simulator's graphics context.
    mp_context = mp.get_context("spawn")
    obs_queue = mp_context.Queue()
    action_queue = mp_context.Queue()

    inference_proc = mp_context.Process(
        target=inference_process,
        args=(obs_queue, action_queue, port, seed, host),
    )
    num_success = 0
    with closing(env), TemporaryDirectory(prefix=".openpi-", dir=output_dir) as staging, ExitStack() as writers:
        try:
            env.start()
            inference_proc.start()

            for ep in range(episodes):
                print(f"Episode {ep + 1}/{episodes}")
                video_dir = Path(staging) / f"episode_{ep:02d}"
                video_dir.mkdir()
                video_writers = {
                    cam_name: writers.enter_context(
                        imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                    )
                    for cam_name in dict.fromkeys(camera_mapping.values())
                }

                env.reset()
                if env_name == "click_mouse":
                    # Align with the dataset's initial pose.
                    # fmt: off
                    initial_action = np.array([
                        -4.4294e-01, 1.3729e-06, 1.5170e00,
                        -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0.263, 0, 0, 0
                    ])
                    # fmt: on
                    for _ in range(30):
                        env.step(initial_action)
                        if env.is_done:
                            break

                _append_video_frames(video_writers, env.get_raw_images())
                timestamp = 0
                actions_buffer: deque[np.ndarray] = deque()
                pending_timestamp: int | None = None
                action_horizon = 0
                password = []

                while not env.is_done:
                    if pending_timestamp is None and (
                        timestamp == 0 or len(actions_buffer) < replan_ratio * action_horizon
                    ):
                        obs_queue.put(env.get_obs())
                        pending_timestamp = timestamp

                    if pending_timestamp is not None:
                        chunk = _take_action_chunk(
                            action_queue, inference_proc, action_dim, wait=timestamp == 0,
                        )
                        if chunk is not None:
                            action_horizon = len(chunk)
                            _merge_action_chunk(chunk, actions_buffer, timestamp - pending_timestamp, dual_arm)
                            pending_timestamp = None

                    pressed_digits = env.step(actions_buffer.popleft()) if actions_buffer else env.stay()
                    if record_pressed_digits and pressed_digits:
                        password.append(pressed_digits)
                    timestamp += 1
                    _append_video_frames(video_writers, env.get_raw_images())

                writers.close()
                num_success += int(env.is_success)
                print("Success!" if env.is_success else "Failed")
                result_suffix = "success" if env.is_success else "failure"
                if record_pressed_digits:
                    password_suffix = "_".join("".join(str(digit) for digit in digits) for digits in password)
                    result_suffix += f"_{password_suffix or 'no_password_input'}"
                video_dir.rename(output_dir / f"episode_{ep:02d}_{result_suffix}")

                if pending_timestamp is not None and ep + 1 < episodes:
                    _take_action_chunk(action_queue, inference_proc, action_dim, wait=True)

        finally:
            if inference_proc.pid is not None:
                if inference_proc.is_alive():
                    obs_queue.put(None)
                inference_proc.join(timeout=2)
                if inference_proc.is_alive():
                    inference_proc.terminate()
                    inference_proc.join(timeout=2)
            obs_queue.cancel_join_thread()
            obs_queue.close()
            action_queue.cancel_join_thread()
            action_queue.close()

    if inference_proc.exitcode not in (0, -signal.SIGTERM):
        raise RuntimeError(f"OpenPI inference worker exited with code {inference_proc.exitcode}")
    print(f"\nSuccess rate: {num_success}/{episodes} ({100 * num_success / episodes:.1f}%)")
    (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()
