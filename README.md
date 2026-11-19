<p align="center">
  <img src="docs/pics/dexjoco_logo.jpg" alt="dexjoco logo" height="110">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.16257"><img src="https://img.shields.io/badge/arXiv-2605.16257-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://dexjoco.github.io/"><img src="https://img.shields.io/badge/GitHub-Page-FF6B00?style=flat-square&logo=github&logoColor=white" alt="Project Homepage"></a>
  <a href="https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot"><img src="https://img.shields.io/badge/🤗%20HF-Dataset-FFD21E?style=flat-square" alt="HF Dataset"></a>
  <a href="https://huggingface.co/DexJoCo/DexJoCo-Pi05"><img src="https://img.shields.io/badge/🤗%20HF-Model-FFD21E?style=flat-square" alt="HF Models"></a>
  <a href="https://huggingface.co/papers/2605.16257"><img src="https://img.shields.io/badge/🤗%20HF-Paper-FFD21E?style=flat-square" alt="HF Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
</p>

---

DexJoCo is a MuJoCo-based simulation benchmark and toolkit for task-oriented
dexterous manipulation. It provides **11 functionally grounded tasks** covering
🛠️ **tool use**, 🤝 **bimanual coordination**, ⏱️ **long-horizon execution**,
and ⚙️ **reasoning**, together with a low-cost teleoperation data collection
system, replayable demonstrations, domain randomization, and OpenPI π0.5 policy
training/evaluation support.

**dexmani_policy (DP / DP3):** use the existing Conda `policy` environment and
the [v1.1 training and evaluation guide](docs/dexmani_policy.md). This path uses
the standalone v1.1 task environments; the OpenPI interfaces and action
ordering described below belong to the teleoperation/OpenPI workflow.

<table width="100%">
  <tr>
    <td align="center" width="20%">
      <img src="docs/pics/photograph_4.png" alt="Bimanual photograph" width="100%">
      <br><sub>Bimanual photograph</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/ipad_2.png" alt="Bimanual unlock iPad" width="100%">
      <br><sub>Bimanual unlock iPad</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/assembly_3.png" alt="Bimanual assembly" width="100%">
      <br><sub>Bimanual assembly</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/hanoi_3.png" alt="Bimanual hanoi" width="100%">
      <br><sub>Bimanual hanoi</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/microwave_2.png" alt="Bimanual microwave" width="100%">
      <br><sub>Bimanual microwave</sub>
    </td>
  </tr>
</table>

<table width="100%">
  <tr>
    <td align="center" width="16.66%">
      <img src="docs/pics/water_plant_4.png" alt="Water plant" width="100%">
      <br><sub>Water plant</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/hammer_3.png" alt="Hammer nail" width="100%">
      <br><sub>Hammer nail</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/glass_1.png" alt="Fold glasses" width="100%">
      <br><sub>Fold glasses</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/tongs_3.png" alt="Pinch tongs" width="100%">
      <br><sub>Pinch tongs</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/bucket_4.png" alt="Pick bucket" width="100%">
      <br><sub>Pick bucket</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/mouse_3.png" alt="Click mouse" width="100%">
      <br><sub>Click mouse</sub>
    </td>
  </tr>
</table>

## Table of Contents

- [Project Layout](#project-layout)
- [Installation](#-installation)
- [Policy Evaluation](#-policy-evaluation)
- [Custom Policy Integration](#-custom-policy-integration)
- [Data Collection](#-data-collection)
- [Demonstration Replay](#-demonstration-replay)
- [Data Format](#data-format)
- [Data Conversion](#data-conversion)
- [Policy Training](#policy-training)
- [Headless Rendering](#headless-rendering)
- [License](#-license)
- [Citation](#-citation)

## Project Layout

- Simulation environments and the OpenPI evaluation client are in [`dexjoco/`](dexjoco/).
- Demonstration recording and replay scripts are in [`scripts/demos/`](scripts/demos/). Hardware setup and bridges are in [`teleoperation/`](teleoperation/).
- OpenPI training and serving code is in [`openpi/`](openpi/).
- Task and evaluation settings are in [`configs/`](configs/).
- The raw demonstration conversion toolkit is in [`dexjoco-data-converter/`](dexjoco-data-converter/).
- Integration guides are in [`docs/`](docs/).

DP/DP3 datasets and run results are stored locally under `datasets/` and `outputs/`.

## 🚀 Installation

Create and activate the DexJoCo environment:

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

OpenPI training and serving environment

```bash
cd openpi
bash install.bash
conda activate openpi
```

## 🤖 Policy Evaluation

Download model checkpoints from
[DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05) or train them
yourself before evaluation.

Keep the downloaded checkpoint paths under `checkpoints/`. Start the server in
the `openpi` environment.

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant \
  --policy.dir ../checkpoints/pi05_dexjoco_ckpt/water_plant
```

Run evaluation from the repository root in the `dexjoco` environment:

```bash
conda activate dexjoco
dexjoco-openpi-eval \
  --config=./configs/rand_obj/water_plant.yaml \
  --seed=0 \
  --port=8000
```

For `rand_full` evaluation, start the server with `--policy.config
water_plant_rand_full` and `--policy.dir
../checkpoints/pi05_dexjoco_rand_full_ckpt/water_plant`. Use the corresponding
evaluation YAML and `--rand-full` on the client.

```bash
dexjoco-openpi-eval \
  --config=./configs/rand_full/water_plant.yaml \
  --seed=0 \
  --port=8000 \
  --rand-full
```

`dexjoco-openpi-eval` options:

| Option                            | Default        | Description                                                        |
| --------------------------------- | -------------- | ------------------------------------------------------------------ |
| `--config PATH`                   | Required       | Evaluation YAML under `configs/rand_obj/` or `configs/rand_full/`. |
| `--seed INT`                      | `0`            | Random seed for NumPy and Python random state.                     |
| `--rand-full`                     | `False`        | Enables the `rand_full` evaluation regime.                         |
| `--randomize-dynamics`            | `False`        | Enables dynamics randomization.                                    |
| `--port INT`                      | `8000`         | OpenPI websocket policy server port.                               |
| `--replan-ratio FLOAT`            | `0.8`          | Request a new chunk when the buffer falls below this fraction of the last response's action count. Accepts values in `(0, 1]`. |
| `--output PATH`                   | Auto-generated | Output directory for videos and success-rate marker files.         |
| `--render-mode {rgb_array,human}` | `rgb_array`    | DexJoCo rendering mode. `rgb_array` is headless.                   |
| `--episodes INT`                  | `50`           | Number of evaluation episodes to run.                              |

Run `dexjoco-openpi-eval --help` for the complete option set.
[OpenPI evaluation CLI](dexjoco/dexjoco_openpi_client/cli/evaluate.py)

## 🔌 Custom Policy Integration

DexJoCo supports custom policy evaluation through the same environment contract
used by the OpenPI client. Observations are collected from the simulator and
passed to a policy for action inference. The resulting actions are executed in
the environment.

Custom integrations should follow the protocol described in
[`docs/custom_policy_integration.md`](docs/custom_policy_integration.md),
including:

- observation fields for camera images, proprioceptive state, and prompts
- action layout conversion from rotation-vector policy actions to quaternion
  environment actions
- chunked action execution and replanning for latency-tolerant inference
- optional multi-frame observation history
- LeRobot-style `async_inference` integration patterns

## 📦 Data Collection

Please refer to the [`teleoperation/`](teleoperation/) directory for the
hardware and software configuration required for teleoperation:

| Component                   | Documentation                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| teleoperation overview      | [`teleoperation/README.md`](teleoperation/README.md)                                                 |
| hardware setup              | [`teleoperation/Teleoperation_System_Tutorial.pdf`](teleoperation/Teleoperation_System_Tutorial.pdf) |
| Vive tracker bridge         | [`teleoperation/vive_bridge`](teleoperation/vive_bridge)                                             |
| Rokoko hand-keypoint bridge | [`teleoperation/rokoko`](teleoperation/rokoko)                                                       |
| GeoRT hand retargeting      | [`teleoperation/GeoRT`](teleoperation/GeoRT)                                                         |

Supported tasks:

| Task           | Setup      | Task Name                 |
| -------------- | ---------- | ------------------------- |
| Unlock iPad    | Bimanual   | `bimanual_unlock_ipad`    |
| Hanoi          | Bimanual   | `bimanual_hanoi`          |
| Assembly       | Bimanual   | `bimanual_assembly`       |
| Microwave cook | Bimanual   | `bimanual_microwave_cook` |
| Photograph     | Bimanual   | `bimanual_photograph`     |
| Hammer nail    | Single-arm | `hammer_nail`             |
| Click mouse    | Single-arm | `click_mouse`             |
| Pick bucket    | Single-arm | `pick_bucket`             |
| Pinch tongs    | Single-arm | `pinch_tongs`             |
| Fold glasses   | Single-arm | `fold_glasses`            |
| Water plant    | Single-arm | `water_plant`             |

Start demonstration recording from the repository root.

```bash
conda activate dexjoco
python scripts/demos/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True \
  --out_dir=./demos
```

The script opens the selected DexJoCo task, records successful teleoperation
episodes, and writes each success as a replayable Zarr episode with camera
videos.

Common `record_demos_zarr.py` options:

| Flag                 | Purpose                                                                       |
| -------------------- | ----------------------------------------------------------------------------- |
| `--exp_name`         | Selects one of the task names.                                                |
| `--successes_needed` | Stops collection after the requested number of successful demos.              |
| `--randomize`        | Enables the `rand_full` visual randomization regime used for data collection. |
| `--show_sim_cameras` | Displays camera streams during interactive collection.                        |
| `--save_depth`       | Saves depth arrays and depth videos alongside RGB videos.                     |
| `--video_fps`        | Sets MP4 playback speed in frames per second.                                |
| `--out_dir`          | Selects the output directory for collected demos.                             |

Use `--camera_screen_effect` for the live display on the camera in
`bimanual_photograph`, or `--ipad_screen_effect` for the unlock transition in
`bimanual_unlock_ipad`. Both flags default to `False`.
[`scripts/demos/record_demos_zarr.py`](scripts/demos/record_demos_zarr.py)

## 🎬 Demonstration Replay

Raw DexJoCo datasets for replay are available from
[`DexJoCo/DexJoCo-Datasets-Raw`](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw).

Replay recorded demonstrations from the repository root.

```bash
conda activate dexjoco
python scripts/demos/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir=./demos \
  --out_dir=./replay_output \
  --randomize=True \
  --restore_state=True
```

Replay sends the recorded quaternion actions directly to the simulator until
success, termination, truncation, or the end of the recording. It restores the
initial scene from `state[0]` by default and can generate `rand_full` visual
variants through camera, lighting, and table texture randomization.

Common `replay_demos_zarr.py` options:

| Flag           | Default           | Purpose                                                                                                 |
| -------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| `--exp_name`   | `water_plant`     | Selects the task used to replay the demonstrations.                                                     |
| `--input_dir`  | `./`              | Directory containing recorded demo folders with `replay.zarr`.                                          |
| `--out_dir`    | `./replay_output` | Output directory for replayed Zarr episodes and videos.                                                 |
| `--randomize`  | `True`            | Enables replay-time `rand_full` visual randomization with preset camera, lighting, and texture changes. |
| `--seed`       | `0`               | Base replay seed; the demo index is added for each input demo.                                          |
| `--restore_state` | `True`         | Requires the recorded state to restore the initial scene. Use `--norestore_state` for a fresh scene.   |
| `--save_failed` | `False`          | Keeps output for replays that do not report success.                                                   |
| `--save_depth` | `False`           | Saves depth arrays and depth videos alongside RGB replay videos.                                        |
| `--video_fps` | `30`              | Sets MP4 playback speed in frames per second.                                                           |

Replay accepts the same task-specific screen-effect flags as recording.
See the script for all supported options.
[`scripts/demos/replay_demos_zarr.py`](scripts/demos/replay_demos_zarr.py)

<a id="data-format"></a>

## 🗂️ Data Format

Each successful demonstration is written as:

```text
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/
  videos/<camera_key>.mp4
  videos/<camera_key>_depth.npz
  videos/<camera_key>_depth.mp4
```

Depth outputs are present when `--save_depth=True`. The NPZ stores camera-axis
depth in metres as float32, with `inf` for pixels on the far clipping plane.
The MP4 is a grayscale preview.

The Zarr replay buffer stores low-dimensional episode data:

| Field           | Description                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `action`        | Recorded float64 actions with wxyz quaternions.                        |
| `action_rotvec` | The same actions with orientations stored as rotation vectors.         |
| `timestamp`     | Seconds from the first frame, using the simulator's 0.02 s control step. |
| `state`         | Proprioception and task state captured before each action.              |

### Policy Training Action and State Layout

For bimanual demonstrations, the recorded action layout is:

```text
[r_pose7, r_hand16, l_pose7, l_hand16]
```

The policy-mode DexJoCo environment expects the flat action layout:

```text
[r_pose7, l_pose7, r_hand16, l_hand16]
```

During OpenPI evaluation,
[`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py`](dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py)
handles the action order conversion automatically. It also converts
rotation-vector actions into the quaternion pose representation used by the
DexJoCo environment.

OpenPI training uses `action_rotvec` as the action target. The raw `action`
field stores quaternion poses with 23 dimensions for single-arm tasks and 46
dimensions for bimanual tasks. `action_rotvec` stores the action layout with 22
dimensions for single-arm tasks and 44 dimensions for bimanual tasks.

The recorded `state` field includes privileged environment state for replay,
such as object poses and table height. Policy training should use only robot
proprioception:

| Setup      | Policy State                                                              |
| ---------- | ------------------------------------------------------------------------- |
| Single-arm | First 23 dimensions: TCP pose and hand joints                             |
| Bimanual   | First 46 dimensions: right TCP pose, left TCP pose, right hand, left hand |

Privileged environment fields should be filtered out before training policy
models.

<a id="data-conversion"></a>

## 🔄 Data Conversion

Use [`dexjoco-data-converter/`](dexjoco-data-converter/) to convert raw DexJoCo
datasets into LeRobot datasets or Zarr replay buffers.

```bash
bash dexjoco-data-converter/install.bash
conda activate dexjoco-dc

dexjoco-dc-single-lerobot \
  --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
  --output ./converted_datasets/lerobot/single/water_plant \
  --language-instruction "Grasp the watering can and apply water to the plant." \
  --selected-data-yaml "{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}" \
  --slice-yaml "{state: [null, 23]}"
```

See [`dexjoco-data-converter/README.md`](dexjoco-data-converter/README.md) for
batch conversion, multi-task merge and configuration files.

<a id="policy-training"></a>

## ⚙️ Policy Training

DexJoCo LeRobot datasets are available from
[`DexJoCo/DexJoCo-Datasets-LeRobot`](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot).

OpenPI π0.5 training support lives under [`openpi/`](openpi). The OpenPI setup
covers two DexJoCo data regimes:

| Regime      | Randomization                                                                  |
| ----------- | ------------------------------------------------------------------------------ |
| `rand_obj`  | Object placement and table height randomization                                |
| `rand_full` | `rand_obj` plus third-person camera, lighting, and table texture randomization |

Training workflow:

1. Install the OpenPI environment with
   [`openpi/install.bash`](openpi/install.bash).
2. Place checkpoints and LeRobot datasets according to
   [`openpi/config.yaml`](openpi/config.yaml).
3. Convert the π0.5 base checkpoint for 44-dimensional bimanual actions with
   [`openpi/scripts/convert_to_action_dim_44_model.py`](openpi/scripts/convert_to_action_dim_44_model.py)
   when training bimanual tasks.
4. Compute normalization statistics with
   [`openpi/scripts/compute_norm_stats.py`](openpi/scripts/compute_norm_stats.py)
   or
   [`openpi/scripts/compute_norm_stats.bash`](openpi/scripts/compute_norm_stats.bash).
5. Launch multiple tmux training jobs with
   [`openpi/scripts/launch_tmux_train.py`](openpi/scripts/launch_tmux_train.py),
   or train a single policy with
   [`openpi/scripts/train.py`](openpi/scripts/train.py).

See [`openpi/README.md`](openpi/README.md) for command examples and checkpoint
layout details.

<a id="headless-rendering"></a>

## 🖥️ Headless Rendering

Headless environments use `policy_mode=True` and `render_mode="rgb_array"`:

```python
TaskConfig.get_environment(
    policy_mode=True,
    render_mode="rgb_array",
    ...
)
```

`policy_mode=True` exposes the policy action interface and disables the
teleoperation wrapper. `render_mode="rgb_array"` selects offscreen rendering for
policy evaluation and automated environment usage.

Interactive teleoperation collection uses `policy_mode=False` and the MuJoCo
viewer, so it does not require the headless configuration.

## 📄 License

DexJoCo-owned code in this repository is released under the
[`MIT License`](LICENSE).

Bundled third-party components and assets retain their separate license terms:

| Component                                                                | License Scope                                                        |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [`teleoperation/GeoRT`](teleoperation/GeoRT)                             | Upstream non-commercial GeoRT license                                |
| [`franka_emika_panda`](dexjoco/dexjoco/sim/envs/xmls/franka_emika_panda) | Apache-2.0                                                           |
| [`wonik_allegro`](dexjoco/dexjoco/sim/envs/xmls/wonik_allegro)           | BSD-2-Clause                                                         |
| [`openpi/`](openpi)                                                      | Apache License, Version 2.0, plus Gemma model terms where applicable |

## 📚 Citation

```bibtex
@misc{wang2026dexjocobenchmarktoolkittaskoriented,
      title={DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
      author={Hanwen Wang and Weizhi Zhao and Xiangyu Wang and Siyuan Huang and He Lin and Boyuan Zheng and Rongtao Xu and Gang Wang and Yao Mu and He Wang and Lue Fan and Hongsheng Li and Zhaoxiang Zhang and Tieniu Tan},
      year={2026},
      eprint={2605.16257},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.16257},
}
```
