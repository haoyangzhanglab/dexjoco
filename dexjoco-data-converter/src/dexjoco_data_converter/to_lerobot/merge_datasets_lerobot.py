import multiprocessing as mp
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..episode_common import find_episodes, find_first_non_static_frame
from ..process_protocol import DoneMsg, ErrorMsg, InitMsg, ProgressBar, ProgressMsg
from ..utils import dict_to_slice, pad_to_dim, terminate_process
from .merge_episode_lerobot import (
    _build_features, _image_size, _load_episode_arrays, _suppress_process_output, _write_episode,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Processing configuration for one input dataset."""

    dataset_id: str
    episode_dirs: list[Path]
    instruction: str
    action_key: str
    state_key: str
    cam_map: OrderedDict[str, str]  # std cam -> raw cam
    slice_cfg: dict[str, slice]


def _build_dataset_specs(
    dataset_dirs: list[Path],
    language_instruction_config: dict[str, str],
    selected_data_config: dict[str, dict],
    slice_config: dict[str, dict[str, slice]],
    bad_episodes_config: dict[str, list[str]] | None = None,
) -> tuple[list[DatasetSpec], list[str], tuple, int]:
    """Build per-dataset processing specs and validate shared LeRobot schema."""
    dataset_specs: list[DatasetSpec] = []
    total_episodes = 0
    cam_key_lists = [
        list(selected_data["cameras"].keys())
        for selected_data in selected_data_config.values()
    ]
    assert len(cam_key_lists) > 0, "No camera config found"
    assert all(cam_keys == cam_key_lists[0] for cam_keys in cam_key_lists), (
        f"Camera key mismatch: {cam_key_lists}"
    )
    standard_cam_names = cam_key_lists[0]

    image_sizes = set()
    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        selected_data = selected_data_config[dataset_id]
        selected_cam_map = OrderedDict(selected_data["cameras"])
        bad_episodes = (
            None if bad_episodes_config is None else bad_episodes_config[dataset_id]
        )
        episode_dirs = find_episodes(dataset_dir, bad_episodes)
        image_sizes.add(_image_size(episode_dirs[0] / "videos", set(selected_cam_map.values())))

        dataset_specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                episode_dirs=episode_dirs,
                instruction=language_instruction_config[dataset_id],
                action_key=selected_data["action"],
                state_key=selected_data["state"],
                cam_map=selected_cam_map,
                slice_cfg=slice_config[dataset_id],
            )
        )
        total_episodes += len(episode_dirs)

    if len(image_sizes) != 1:
        raise ValueError(f"Expected one image size across datasets, got {sorted(image_sizes)}")
    image_size = image_sizes.pop()
    return dataset_specs, standard_cam_names, image_size, total_episodes


def merge_datasets_worker(
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction_config: dict[str, str],
    slice_config: dict[str, dict[str, slice]],
    selected_data_config: dict[str, dict],
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_config: dict[str, list[str]] | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a multi-dataset LeRobot merge with optional output suppression.

    Args:
        dataset_dirs: Dataset roots to merge in order.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction_config: Per-dataset task instructions.
        slice_config: Per-dataset slices keyed by selected data key.
        selected_data_config: Per-dataset mapping for action, state, and cameras.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_config: Per-dataset episode names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.
    """
    output_ctx = _suppress_process_output() if silent else nullcontext()
    with output_ctx:
        _merge_datasets_worker_impl(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            dataset_name=dataset_name,
            message_queue=message_queue,
            language_instruction_config=language_instruction_config,
            slice_config=slice_config,
            selected_data_config=selected_data_config,
            target_action_dim=target_action_dim,
            target_state_dim=target_state_dim,
            bad_episodes_config=bad_episodes_config,
            skip_static_frames=skip_static_frames,
        )


def _merge_datasets_worker_impl(
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_name: str,
    message_queue: mp.Queue,
    language_instruction_config: dict[str, str],
    slice_config: dict[str, dict[str, slice]],
    selected_data_config: dict[str, dict],
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_config: dict[str, list[str]] | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Merge several datasets into one unified LeRobot dataset.

    Args:
        dataset_dirs: Dataset roots to merge in order.
        output_dir: Destination LeRobot dataset root.
        dataset_name: Name used in progress messages.
        message_queue: Multiprocessing queue used to emit worker status messages.
        language_instruction_config: Per-dataset task instructions.
        slice_config: Per-dataset slices keyed by selected data key.
        selected_data_config: Per-dataset mapping for action, state, and cameras.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_config: Per-dataset episode names excluded from the merge.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    try:
        dataset_specs, standard_cam_names, image_size, total_episodes = (
            _build_dataset_specs(
                dataset_dirs=dataset_dirs,
                language_instruction_config=language_instruction_config,
                selected_data_config=selected_data_config,
                slice_config=slice_config,
                bad_episodes_config=bad_episodes_config,
            )
        )

        message_queue.put(
            InitMsg(dataset_name=dataset_name, total_episodes=total_episodes)
        )
        features = _build_features(
            standard_cam_names,
            image_size=image_size,
            action_shape=(target_action_dim,),
            state_shape=(target_state_dim,),
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
    except Exception as e:
        message_queue.put(ErrorMsg(dataset_name=dataset_name, error=str(e)))
        raise

    total_steps = 0
    total_truncated_frames = 0
    try:
        for spec in dataset_specs:
            selected_action_key = spec.action_key
            selected_state_key = spec.state_key
            selected_cam_map = spec.cam_map
            slice_cfg = spec.slice_cfg

            for ep_dir in spec.episode_dirs:
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
                    # Enforce a fixed action/state schema across mixed datasets by right-padding with zeros.
                    episode_data[selected_action_key] = pad_to_dim(
                        episode_data[selected_action_key], target_action_dim
                    )
                    episode_data[selected_state_key] = pad_to_dim(
                        episode_data[selected_state_key], target_state_dim
                    )

                    _write_episode(
                        dataset, videos_dir, episode_data[selected_action_key], episode_data[selected_state_key],
                        selected_cam_map, spec.instruction, start_idx,
                    )
                    total_steps += n_steps

                    message_queue.put(
                        ProgressMsg(
                            dataset_name=dataset_name,
                            episode_name=f"{spec.dataset_id}/{ep_dir.name}",
                        )
                    )
                except Exception as e:
                    message_queue.put(
                        ErrorMsg(
                            dataset_name=dataset_name,
                            error=str(e),
                            episode_name=f"{spec.dataset_id}/{ep_dir.name}",
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


def merge_datasets(
    output: Path,
    dataset_paths_cfg_path: Path,
    language_instruction_cfg_path: Path,
    selected_data_cfg_path: Path,
    slice_cfg_path: Path,
    target_action_dim: int,
    target_state_dim: int,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Run a multi-dataset LeRobot merge from YAML configuration files.

    Args:
        output: Destination LeRobot dataset root.
        dataset_paths_cfg_path: YAML file containing dataset root paths.
        language_instruction_cfg_path: YAML file containing per-dataset task
            instructions.
        selected_data_cfg_path: YAML file mapping each dataset to selected action,
            state, and camera keys.
        slice_cfg_path: YAML file containing per-dataset slice specifications.
        target_action_dim: Required action feature dimension after padding.
        target_state_dim: Required state feature dimension after padding.
        bad_episodes_cfg_path: Optional YAML file containing per-dataset excluded
            episodes.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside the worker.

    Returns:
        None.

    Raises:
        RuntimeError: If the output directory is not empty or the worker reports a
            failure.
        AssertionError: If configured dataset paths do not exist.
        Exception: If the worker process exits with a non-zero code.
    """
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    # Load all merge controls from config files so one worker can process all datasets sequentially.
    with open(dataset_paths_cfg_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    dataset_dirs = [Path(p) for p in dataset_paths]
    assert all(p.exists() for p in dataset_dirs), "Some datasets path do not exist"
    with open(language_instruction_cfg_path, "r") as f:
        language_instruction_config = yaml.safe_load(f)
    with open(selected_data_cfg_path, "r") as f:
        selected_data_config = yaml.safe_load(f)
    with open(slice_cfg_path, "r") as f:
        raw_slice_config = yaml.safe_load(f)
    slice_config = {k: dict_to_slice(v) for k, v in raw_slice_config.items()}
    if bad_episodes_cfg_path is not None:
        with open(bad_episodes_cfg_path, "r") as f:
            bad_episodes_config = yaml.safe_load(f)
    else:
        bad_episodes_config = None

    dataset_name = output.stem
    msg_que: mp.Queue = mp.Queue()
    worker = mp.Process(
        target=merge_datasets_worker,
        kwargs={
            "dataset_dirs": dataset_dirs,
            "output_dir": output,
            "dataset_name": dataset_name,
            "message_queue": msg_que,
            "language_instruction_config": language_instruction_config,
            "slice_config": slice_config,
            "selected_data_config": selected_data_config,
            "target_action_dim": target_action_dim,
            "target_state_dim": target_state_dim,
            "bad_episodes_config": bad_episodes_config,
            "skip_static_frames": skip_static_frames,
            "silent": silent,
        },
    )
    worker.start()

    pbar = ProgressBar(idx=0, dataset_name=dataset_name)
    try:
        while True:
            try:
                msg = msg_que.get(timeout=1.0)
            except Empty:
                if not worker.is_alive() and msg_que.empty():
                    break
                continue

            pbar.update(msg)
    finally:
        terminate_process(worker, process_name=dataset_name)
        msg_que.cancel_join_thread()
        msg_que.close()
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
