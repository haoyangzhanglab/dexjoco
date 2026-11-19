#!/usr/bin/env python3
"""Record successful teleoperation episodes with synchronized camera videos."""

from contextlib import ExitStack, closing
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags
from tqdm import tqdm

from dexjoco.data.demo_export import write_demo
from dexjoco.data.depth_capture import collect_depth_frames
from dexjoco.tasks.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_enum("exp_name", "water_plant", list(CONFIG_MAPPING), "Task name.")
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.", lower_bound=1)
flags.DEFINE_bool("show_sim_cameras", True, "Show simulation cameras")
flags.DEFINE_integer(
    "max_steps",
    0,
    "Stop after this many environment steps; 0 means run until successes_needed",
    lower_bound=0,
)
flags.DEFINE_string("out_dir", "./", "Output base directory for zarr and videos")
flags.DEFINE_integer("video_fps", 30, "Playback FPS for saved MP4 videos", lower_bound=1)
flags.DEFINE_enum(
    "render_mode",
    "human",
    ["human", "rgb_array"],
    "MuJoCo render mode for the task environment",
)
flags.DEFINE_bool(
    "randomize",
    False,
    "Enable environment randomization when the selected task supports it",
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


def main(_argv):
    task_id = FLAGS.exp_name
    config = CONFIG_MAPPING[task_id]()
    env_kwargs = {}
    if task_id == "bimanual_photograph":
        env_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    elif task_id == "bimanual_unlock_ipad":
        env_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect

    base_out = Path(FLAGS.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with ExitStack() as stack:
        env = stack.enter_context(closing(config.get_environment(
            render_mode=FLAGS.render_mode,
            randomize=FLAGS.randomize,
            **env_kwargs,
        )))
        progress = stack.enter_context(tqdm(total=FLAGS.successes_needed))
        obs, _ = env.reset()
        idle_action = np.zeros(env.action_space.shape)
        wrist_windows = {}
        if FLAGS.show_sim_cameras:
            for key in ("wrist", "wrist_left", "wrist_right"):
                if key in obs:
                    window = key.replace("_", " ").title()
                    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                    stack.callback(cv2.destroyWindow, window)
                    wrist_windows[key] = window

        trajectory: list[dict] = []
        return_sum = 0.0
        total_steps = 0
        while len(saved) < FLAGS.successes_needed:
            for key, window in wrist_windows.items():
                frame = cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR)
                cv2.imshow(window, cv2.resize(frame, None, fx=0.75, fy=0.75))
            if wrist_windows and (cv2.waitKey(1) & 0xFF) == ord("r"):
                print("[Keyboard] Reset triggered")
                trajectory.clear()
                return_sum = 0.0
                obs, _ = env.reset()
                continue

            transition = {"observations": obs}
            if FLAGS.save_depth:
                transition["depth"] = collect_depth_frames(env, obs)
            obs, reward, terminated, truncated, info = env.step(idle_action)
            total_steps += 1
            return_sum += reward
            progress.set_description(f"Return: {return_sum:.3f}")

            manual_reset = info.get("manual_reset", False)
            if manual_reset:
                print("[Teleop] Manual reset triggered; dropping current trajectory.")
            else:
                action = info["intervene_action"]
                if isinstance(action, dict):
                    action = np.concatenate((action["right"], action["left"]))
                transition["actions"] = np.asarray(action, dtype=np.float64)
                trajectory.append(transition)

                if info["succeed"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                    demo_dir = base_out / f"{task_id}_demo_{len(saved) + 1}_{timestamp}"
                    write_demo(demo_dir, trajectory, env.unwrapped.control_dt, FLAGS.video_fps)
                    saved.append(demo_dir)
                    print(f"[Saved] {demo_dir}")
                    progress.update(1)

            if FLAGS.max_steps and total_steps >= FLAGS.max_steps:
                print(f"Reached max_steps={FLAGS.max_steps}; stopping run.")
                break

            if manual_reset or terminated or truncated or info["succeed"]:
                trajectory.clear()
                return_sum = 0.0
                if len(saved) < FLAGS.successes_needed:
                    obs, _ = env.reset()

    print(f"Collected {len(saved)} successful demos.")
    for path in saved:
        print("  ", path)


if __name__ == "__main__":
    app.run(main)
