import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from queue import Empty
from typing import Any, cast

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import yaml
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from imageio.core.format import Format
from zarr import Array, Group

from ..episode_common import find_episodes, find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, normalize_array_shape, terminate_process


def _load_episode_arrays(
    zarr_path: Path, selected_keys: list[str]
) -> dict[str, np.ndarray]:
    """Load selected episode arrays from a replay Zarr store.

    Args:
        zarr_path: Path to an episode ``replay.zarr`` directory.
        selected_keys: Data keys to load from the Zarr ``data`` group.

    Returns:
        Mapping from selected key to a normalized ``float32`` NumPy array.
    """
    root = cast(Group, zarr.open(str(zarr_path), mode="r"))
    data_group = cast(Group, root["data"])

    episode_data: dict[str, np.ndarray] = {}
    for key in selected_keys:
        raw = np.asarray(cast(Array, data_group[key])[:], dtype=np.float32)
        episode_data[key] = normalize_array_shape(raw)
    return episode_data


def _image_size(videos_dir: Path, raw_cameras: set[str]) -> tuple[int, int]:
    sizes = set()
    for camera in sorted(raw_cameras):
        video = videos_dir / f"{camera}.mp4"
        height, width, _ = iio.improps(video, index=0).shape
        sizes.add((height, width))
    if len(sizes) != 1:
        raise ValueError(f"{videos_dir}: expected selected RGB videos of one size, got {sorted(sizes)}")
    return sizes.pop()


def _write_episode(
    dataset: LeRobotDataset,
    videos_dir: Path,
    actions: np.ndarray,
    states: np.ndarray,
    camera_map: dict[str, str],
    instruction: str,
    start_idx: int,
) -> None:
    with ExitStack() as stack:
        frames_by_camera = {}
        for raw_camera in dict.fromkeys(camera_map.values()):
            reader = cast(
                Format.Reader,
                stack.enter_context(imageio.get_reader(videos_dir / f"{raw_camera}.mp4")),
            )
            frames_by_camera[raw_camera] = islice(reader.iter_data(), start_idx, None)
        aligned = zip(actions, states, *frames_by_camera.values(), strict=True)
        for action, state, *frames in aligned:
            raw_images = dict(zip(frames_by_camera, frames, strict=True))
            frame_data: dict[str, Any] = {"task": instruction, ACTION: action, OBS_STATE: state}
            for camera, raw_camera in camera_map.items():
                frame_data[f"{OBS_IMAGES}.{camera}"] = raw_images[raw_camera]
            dataset.add_frame(frame_data)
    dataset.save_episode()


def _build_features(
    camera_names: list[str],
    image_size: tuple,
    action_shape: tuple,
    state_shape: tuple,
) -> dict[str, Any]:
    """Build a LeRobot feature schema for images, action, and state.

    Args:
        camera_names: Camera names exposed as observation image keys.
        image_size: Frame size as ``(height, width)``.
        action_shape: Per-frame action tensor shape.
        state_shape: Per-frame state tensor shape.

    Returns:
        Feature specification accepted by ``LeRobotDataset.create``.
    """
    features: dict[str, Any] = {}

    for cam_name in camera_names:
        features[f"{OBS_IMAGES}.{cam_name}"] = {
            "dtype": "video",
            "shape": (image_size[0], image_size[1], 3),
            "names": ["height", "width", "channel"],
        }

    features.update({
        ACTION: {"dtype": "float32", "shape": action_shape},
        OBS_STATE: {"dtype": "float32", "shape": state_shape},
    })

    return features


@contextmanager
def _suppress_process_output():
    """Temporarily redirect process stdout and stderr to ``os.devnull``.

    Args:
        None.

    Yields:
        None while stdout and stderr are suppressed.
    """
    stdout = sys.stdout
    stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = stdout
        sys.stderr = stderr
        devnull.close()


def merge_episode_worker(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    message_queue: Any,
    language_instruction: str,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a single-dataset LeRobot merge with optional worker output suppression.

    Args:
        input_dir: Dataset directory containing episode subdirectories.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction: Task instruction written to each output frame.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.
    """
    output_ctx = _suppress_process_output() if silent else nullcontext()
    with output_ctx:
        _merge_episode_worker_impl(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            message_queue=message_queue,
            language_instruction=language_instruction,
            slice_cfg=slice_cfg,
            selected_data=selected_data,
            bad_episodes=bad_episodes,
            skip_static_frames=skip_static_frames,
        )


def _merge_episode_worker_impl(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction: str,
    slice_cfg: dict[str, slice],
    selected_data: dict,
    bad_episodes: list[str] | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Merge one dataset of episode folders into a LeRobot dataset.

    Args:
        input_dir: Dataset directory containing episode subdirectories.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Dataset name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction: Task instruction written to each output frame.
        slice_cfg: Mapping from selected data key to slice applied after truncation.
        selected_data: Mapping with selected action, state, and camera keys.
        bad_episodes: Episode directory names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    try:
        selected_action_key: str = selected_data["action"]
        selected_state_key: str = selected_data["state"]
        selected_cam_map: OrderedDict[str, str] = OrderedDict(selected_data["cameras"])
        selected_raw_cams = set(selected_cam_map.values())

        episode_dirs = find_episodes(input_dir, bad_episodes)
        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=len(episode_dirs))
        )

        image_size = _image_size(episode_dirs[0] / "videos", selected_raw_cams)
        first_episode_data = _load_episode_arrays(
            episode_dirs[0] / "replay.zarr",
            selected_keys=[selected_action_key, selected_state_key],
        )
        for k, v in slice_cfg.items():
            first_episode_data[k] = first_episode_data[k][:, v]

        features = _build_features(
            list(selected_cam_map.keys()),
            image_size=image_size,
            action_shape=tuple(first_episode_data[selected_action_key].shape[1:]),
            state_shape=tuple(first_episode_data[selected_state_key].shape[1:]),
        )

        # lerobot will automatically create data folder
        dataset = LeRobotDataset.create(
            repo_id="local_repo",
            fps=30,
            features=features,
            root=output_dir,
            image_writer_threads=4,
            streaming_encoding=True,
            # ! 0 represents infinite queue size
            # ! lerobot streaming video encoder drops frames when the queue is full
            encoder_queue_maxsize=0,
        )

        total_steps = 0
        total_truncated_frames = 0
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    try:
        for ep_dir in episode_dirs:
            try:
                zarr_file = ep_dir / "replay.zarr"
                videos_dir = ep_dir / "videos"

                episode_data = _load_episode_arrays(
                    zarr_file,
                    selected_keys=[selected_action_key, selected_state_key],
                )
                n_steps = episode_data[selected_action_key].shape[0]

                if skip_static_frames:
                    start_idx = find_first_non_static_frame(
                        episode_data[selected_action_key]
                    )
                    if start_idx < n_steps and np.all(
                        episode_data[selected_action_key][start_idx] == 0
                    ):
                        start_idx += 1
                    total_truncated_frames += start_idx
                    for key in episode_data:
                        episode_data[key] = episode_data[key][start_idx:]
                    n_steps -= start_idx
                else:
                    start_idx = 0
                for k, v in slice_cfg.items():
                    episode_data[k] = episode_data[k][:, v]

                _write_episode(
                    dataset, videos_dir, episode_data[selected_action_key], episode_data[selected_state_key],
                    selected_cam_map, language_instruction, start_idx,
                )
                total_steps += n_steps

                message_queue.put(
                    ProgressMsg(dataset_name=dataset_name, episode_name=ep_dir.name)
                )
            except Exception as e:
                message_queue.put(
                    ErrorMsg(
                        dataset_name=dataset_name,
                        error=str(e),
                        episode_name=ep_dir.name,
                    )
                )
                raise
    finally:
        dataset.finalize()

    statistics_msg = (
        f"Output directory: {output_dir}\n"
        f"Total steps: {total_steps}\n"
        f"Total frames truncated: {total_truncated_frames}"
    )
    message_queue.put(DoneMsg(dataset_name=dataset_name, statistics_msg=statistics_msg))


def merge_episode(
    input: Path,
    output: Path,
    language_instruction: str,
    selected_data_yaml: str,
    slice_yaml: str,
    bad_episodes_yaml: str | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a single-dataset LeRobot merge in a worker process.

    Args:
        input: Dataset directory containing episode subdirectories.
        output: Destination LeRobot dataset root.
        language_instruction: Task instruction written to each output frame.
        selected_data_yaml: YAML string containing selected action, state, and camera keys.
        slice_yaml: YAML string containing slice configuration keyed by selected data key.
        bad_episodes_yaml: YAML string containing episode directory names excluded
            from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty or the worker reports a
            failure.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    selected_data = yaml.safe_load(selected_data_yaml)
    slice_spec = yaml.safe_load(slice_yaml)
    bad_episodes = (
        None if bad_episodes_yaml is None else yaml.safe_load(bad_episodes_yaml)
    )
    assert isinstance(selected_data, dict), (
        "selected_data_yaml must be a YAML dict string"
    )
    assert isinstance(slice_spec, dict), "slice_yaml must be a YAML dict string"
    assert bad_episodes is None or isinstance(bad_episodes, list), (
        "bad_episodes_yaml must be a YAML list string"
    )

    dataset_name = input.name

    queue: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_episode_worker,
        kwargs={
            "input_dir": input,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": queue,
            "language_instruction": language_instruction,
            "slice_cfg": dict_to_slice(slice_spec),
            "selected_data": selected_data,
            "bad_episodes": bad_episodes,
            "skip_static_frames": skip_static_frames,
            "silent": silent,
        },
    )
    worker.start()

    pbar = ProgressBar(idx=0, dataset_name=dataset_name)
    try:
        while True:
            try:
                msg = queue.get(timeout=1.0)
            except Empty:
                if not worker.is_alive() and queue.empty():
                    break
                continue

            pbar.update(msg)
    finally:
        terminate_process(worker, process_name=dataset_name)

        queue.cancel_join_thread()
        queue.close()
        pbar.close()

    state = pbar.state
    match state.status:
        case "failed":
            raise RuntimeError(state.last_error)
        case "done":
            if state.done_msg is not None:
                print(state.done_msg)
        case "running":
            raise RuntimeError(f"{dataset_name} exited without sending done message")
