from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium import spaces

from dexjoco.actions import quaternion_to_rotvec
from dexjoco.env_factory import TASKS, make_env
from dexjoco.policy_env import DexJoCoEnv, depth_to_point_cloud


def test_policy_recipes(monkeypatch):
    from dexmani_policy.training.config import load_config

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("DEXJOCO_ROOT", str(root))
    recipes = [
        load_config(
            f"{method}_dexjoco", ["task_name=bimanual_hanoi", "dataloader.batch_size=8"],
            config_dir="configs/dexmani_policy",
        )
        for method in ("dp", "dp3")
    ]
    for cfg in recipes:
        assert cfg.agent.state_key == "proprio"
        assert cfg.env_runner.task_name == "bimanual_hanoi"
        assert cfg.dataloader.batch_size == 8
        assert cfg.zarr_path == str(root / "datasets/policy/bimanual_hanoi.zarr")
        assert "randomize_model_id" not in cfg.env_runner.env_kwargs
    # Checkpoint selection requires identical environment settings across methods.
    assert recipes[0].env_runner == recipes[1].env_runner


def test_smoke_preserves_caller_paths(monkeypatch):
    from dexmani_policy.test_tools import smoke_test

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("DEXJOCO_ROOT", str(root))
    calls = []

    def check_run(cfg, directory, *, max_updates, resume_updates):
        assert Path.cwd() == root
        assert cfg.agent.state_key == "proprio"
        assert Path(cfg.dataset.zarr_path).resolve() == root / "datasets/smoke/water_plant.zarr"
        assert directory == root / "outputs/smoke/path_test"
        assert (max_updates, resume_updates) == (4, 2)
        calls.append(directory)
        return 4, 6

    monkeypatch.setattr(smoke_test, "run_smoke", check_run)
    smoke_test.main([
        "dp_dexjoco", "--config-dir", "configs/dexmani_policy",
        "--output-dir", "outputs/smoke/path_test",
        "--override", "zarr_path=datasets/smoke/water_plant.zarr",
    ])
    assert len(calls) == 1


@pytest.mark.parametrize("dtype, frames, error", [
    ("float64", [1, 0], None),
    ("float32", [1, 0], "float64 quaternion actions"),
    ("float64", [0, 0], "frame indices"),
])
def test_read_episode_action_precision_and_order(tmp_path, dtype, frames, error):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from dexjoco.data.lerobot import read_episodes

    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    # Metadata alone cannot guarantee the actual Parquet action precision.
    (tmp_path / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "features": {"action": {"dtype": "float64"}},
    }))
    (tmp_path / "replay_initial_values.jsonl").write_text(json.dumps({
        "episode_index": 0, "replay_initial_values": {"table_delta_height": 0.0},
    }) + "\n")
    actions = np.arange(46, dtype=dtype).reshape(2, 23)
    pq.write_table(pa.table({
        "episode_index": [0, 0], "frame_index": frames,
        "action": pa.array(actions.tolist(), type=pa.list_(pa.from_numpy_dtype(dtype))),
    }), tmp_path / "data/episode.parquet")
    if error is not None:
        with pytest.raises(ValueError, match=error):
            list(read_episodes(tmp_path))
    else:
        [(index, actual, initial)] = read_episodes(tmp_path)
        assert index == 0 and initial == {"table_delta_height": 0.0}
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, actions[::-1])


def test_metric_depth_and_pixel_centers():
    model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <visual><global offwidth="80" offheight="48"/><quality offsamples="0"/></visual>
          <worldbody>
            <geom type="plane" size="10 10 .1" rgba=".3 .5 .7 1"/>
            <camera name="ego" pos="0 0 2" fovy="60"/>
          </worldbody>
        </mujoco>
    ''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=48, width=80) as renderer:
        renderer.update_scene(data, camera="ego")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    cloud = depth_to_point_cloud(
        rgb, depth, fovy=60, position=data.cam_xpos[0],
        rotation=data.cam_xmat[0].reshape(3, 3),
        far=model.vis.map.zfar * model.stat.extent, num_points=48 * 80,
    ).reshape(48, 80, 6)
    np.testing.assert_allclose(depth, 2, atol=2e-5)
    np.testing.assert_allclose(cloud[..., 2], 0, atol=2e-5)
    pixel_size = 2 * 2 * np.tan(np.deg2rad(30)) / 48
    np.testing.assert_allclose(cloud[0, 0, :2], [-39.5 * pixel_size, 23.5 * pixel_size], atol=2e-5)
    np.testing.assert_allclose(cloud[-1, -1, :2], [39.5 * pixel_size, -23.5 * pixel_size], atol=2e-5)
    np.testing.assert_allclose(cloud[..., 3:], rgb / 255, atol=1e-7)


@pytest.mark.parametrize("modalities", [
    ("proprio", "rgb"), ("proprio", "point_cloud"), ("point_cloud", "proprio"),
])
def test_observation_modalities(modalities):
    with closing(DexJoCoEnv(
        "water_plant", sensor_modalities=modalities, render_width=64, render_height=64,
    )) as env:
        observation, _ = env.reset(seed=7)
        assert isinstance(env.observation_space, spaces.Dict)
        assert tuple(env.observation_space.spaces) == modalities
        assert tuple(observation) == modalities
        assert env.observation_space.contains(observation)


@pytest.mark.parametrize("task", TASKS)
def test_native_observation_and_step(task):
    kwargs = dict(render_width=64, render_height=64, offsamples=0)
    with (
        closing(make_env(task, action_format="rotvec", **kwargs)) as native,
        closing(DexJoCoEnv(task, record_video=True, **kwargs)) as env,
    ):
        raw, _ = native.reset(seed=7)
        obs, _ = env.reset(seed=7)
        assert env.action_cnt == 0
        assert env.metadata == {"render_modes": ["rgb_array"], "render_fps": 50}
        for step in range(3):
            np.testing.assert_array_equal(obs["proprio"], raw["agent_pos"])
            np.testing.assert_array_equal(obs["rgb"], raw["pixels"]["ego"])
            assert env.observation_space.contains(obs)
            assert np.isfinite(obs["point_cloud"]).all()
            # Reproject world points to the RGB image to check axes, pose and alignment.
            camera = env.camera_id
            rotation = env.env.data.cam_xmat[camera].reshape(3, 3)
            xyz = (obs["point_cloud"][:, :3] - env.env.data.cam_xpos[camera]) @ rotation
            focal = 32 / np.tan(np.deg2rad(env.env.model.cam_fovy[camera]) / 2)
            u = np.rint(xyz[:, 0] / -xyz[:, 2] * focal + 31.5).astype(int)
            v = np.rint(xyz[:, 1] / xyz[:, 2] * focal + 31.5).astype(int)
            np.testing.assert_allclose(
                obs["point_cloud"][:, 3:], obs["rgb"][v, u] / 255, atol=1e-7,
            )
            if step == 2:
                break
            action = quaternion_to_rotvec(
                raw["agent_pos"][None], bimanual=task.startswith("bimanual_"),
            )[0]
            raw, reward, terminated, truncated, info = native.step(action)
            obs, actual_reward, actual_terminated, actual_truncated, actual_info = env.step(action)
            assert (reward, terminated, truncated) == (
                actual_reward, actual_terminated, actual_truncated,
            )
            assert actual_info["success"] == actual_info["success_condition"] == info["is_success"]
        assert env.action_cnt == native.state_machine.step_count == 2
        assert env.dt == 0.02 and env.video_fps == 50
        assert env.get_video().shape == (3, 64, 64, 3)
        np.testing.assert_array_equal(env.render(), raw["pixels"]["ego"])
        env.reset(seed=7)
        assert env.action_cnt == 0
        assert env.get_video().shape == (1, 64, 64, 3)
