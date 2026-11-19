"""Shared observations for demonstration conversion and policy rollouts."""

from typing import cast

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from dexjoco.common import CONTROL_TIMESTEP
from dexjoco.env_factory import TaskEnv, make_env


def _policy_info(info: dict) -> dict:
    success = bool(info.pop("is_success"))
    info.update(success=success, success_condition=success)
    return info


def depth_to_point_cloud(rgb, depth, *, fovy, position, rotation, far, num_points):
    """Metric depth to world XYZ (meters) and RGB [0, 1], at pixel centers."""
    height, width = depth.shape
    valid = np.flatnonzero(np.isfinite(depth) & (depth > 0) & (depth < far * (1 - 1e-5)))
    if not len(valid):
        raise ValueError("The ego camera has no valid depth pixels")
    # Uniform pixel selection is deterministic; repeat pixels when fewer are visible.
    indices = valid[np.linspace(0, len(valid) - 1, num_points).astype(np.int64)]
    v, u = np.unravel_index(indices, depth.shape)
    z = depth[v, u]
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    camera_points = np.column_stack((
        (u + 0.5 - width / 2) * z / focal,
        -(v + 0.5 - height / 2) * z / focal,
        -z,
    ))
    # MuJoCo camera axes: +x right, +y up, -z forward.
    world_points = camera_points @ rotation.T + position
    return np.column_stack((world_points, rgb[v, u] / 255.0)).astype(np.float32)


class DexJoCoEnv(gym.Wrapper):
    env: TaskEnv

    def __init__(
        self, task_name, *, sensor_modalities=("proprio", "rgb", "point_cloud"),
        action_format="rotvec", render_width=224, render_height=224, num_points=4096,
        randomize=False, randomize_dynamics=False, offsamples=0, record_video=False,
    ):
        super().__init__(make_env(
            task_name, render_mode="rgb_array", action_format=action_format,
            render_width=render_width, render_height=render_height,
            randomize=randomize, randomize_dynamics=randomize_dynamics, offsamples=offsamples,
        ))
        self.num_points = num_points
        self.camera = "back" if task_name.startswith("bimanual_") else "front"
        self.camera_id = self.env.model.camera(self.camera).id
        self.dt = CONTROL_TIMESTEP
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": round(1 / self.dt)}
        self.record_video = record_video
        self.video_frames = []
        native_spaces = cast(spaces.Dict, self.env.observation_space)
        observation_spaces = {
            "proprio": native_spaces["agent_pos"],
            "rgb": cast(spaces.Dict, native_spaces["pixels"])["ego"],
            "point_cloud": spaces.Box(-np.inf, np.inf, (num_points, 6), dtype=np.float32),
        }
        self.observation_space = spaces.Dict([
            (key, observation_spaces[key]) for key in sensor_modalities
        ])

    @property
    def action_cnt(self) -> int:
        return self.env.state_machine.step_count

    def observation(self, raw):
        self.rgb = raw["pixels"]["ego"]
        observation = {"proprio": raw["agent_pos"], "rgb": self.rgb}
        observation_spaces = cast(spaces.Dict, self.observation_space).spaces
        if "point_cloud" in observation_spaces:
            # Reuse the RGB renderer and scene options without advancing visual effects again.
            renderer = cast(mujoco.Renderer, self.env.renderer)
            renderer.enable_depth_rendering()
            try:
                depth = self.env.render_camera(self.camera)
            finally:
                renderer.disable_depth_rendering()
            model, data = self.env.model, self.env.data
            observation["point_cloud"] = depth_to_point_cloud(
                self.rgb, depth, fovy=model.cam_fovy[self.camera_id],
                position=data.cam_xpos[self.camera_id],
                rotation=data.cam_xmat[self.camera_id].reshape(3, 3),
                far=model.vis.map.zfar * model.stat.extent, num_points=self.num_points,
            )
        return {key: observation[key] for key in observation_spaces}

    def reset(self, *, seed=None, options=None):
        self.video_frames.clear()
        raw, info = self.env.reset(seed=seed, options=options)
        return self.observation(raw), _policy_info(info)

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation(raw)
        if self.record_video:
            self.video_frames.append(self.rgb)
        return observation, reward, terminated, truncated, _policy_info(info)

    def render(self):
        return self.rgb

    def get_video(self):
        if not self.video_frames:
            return None
        video = np.stack(self.video_frames)
        self.video_frames.clear()
        return video
