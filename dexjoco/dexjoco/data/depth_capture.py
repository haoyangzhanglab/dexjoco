"""Capture camera-axis depth in metres and export a grayscale video preview."""

from pathlib import Path

import numpy as np

from dexjoco.data.video_writer import write_video


def _linearise_depth(z_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.full(z_buffer.shape, np.inf, dtype=np.float32)
    # Gymnasium's offscreen viewer returns the OpenGL z-buffer. A value of 1
    # indicates a pixel left on the far clipping plane.
    visible = (z_buffer >= 0) & (z_buffer < 1)
    depth[visible] = near * far / (near + (1 - z_buffer[visible]) * (far - near))
    return depth


def collect_depth_frames(env, observation: dict) -> dict[str, np.ndarray]:
    raw_env = env.unwrapped
    if raw_env._viewer is None:
        # WaterPlant renders wrist first with a separate viewer.
        viewer = raw_env._mj_viewer
        camera_ids = (raw_env._wrist_camera_id, raw_env._front_camera_id)
    else:
        viewer = raw_env._viewer
        camera_ids = raw_env.camera_id

    model = raw_env.model
    near = model.vis.map.znear * model.stat.extent
    far = model.vis.map.zfar * model.stat.extent
    camera_keys = (key for key in observation if key != "state")
    return {
        key: _linearise_depth(viewer.render("depth_array", camera_id=camera_id), near, far)
        for key, camera_id in zip(camera_keys, camera_ids, strict=True)
    }


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    valid_depths = depth[finite]
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if valid_depths.size:
        low, high = valid_depths.min(), valid_depths.max()
        if high > low:
            gray[finite] = ((valid_depths - low) / (high - low) * 255).astype(np.uint8)
    return np.broadcast_to(gray[..., None], (*gray.shape, 3))


def write_depth_outputs(depth_frames_per_camera: dict, videos_dir: Path, video_fps: int) -> None:
    for key, frames in depth_frames_per_camera.items():
        depth = np.stack(frames)
        np.savez_compressed(videos_dir / f"{key}_depth.npz", depth=depth)
        write_video(videos_dir / f"{key}_depth.mp4", _depth_preview(depth), video_fps)
