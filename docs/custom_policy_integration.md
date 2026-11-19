# Custom Policy Integration

This document describes how to adapt a custom policy to DexJoCo and how to
evaluate it against the benchmark environments.

This guide covers the teleoperation/OpenPI environment interfaces in `sim/`
and `tasks/`. The OpenPI client in
[`dexjoco/dexjoco_openpi_client`](../dexjoco/dexjoco_openpi_client) is the
reference for this path. The LeRobot reference under
[`docs/reference_code`](reference_code) illustrates `async_inference` reuse.

For DP/DP3 with the standalone v1.1 environments, use the
[dexmani_policy guide](dexmani_policy.md). Its proprioception layout and native
rotvec interface differ from the quaternion wrapper described here.

## Policy Protocol

### Observation

A policy should consume the observation fields defined by the task config. The
core fields are:

- camera images mapped through `camera_mapping`
- `state`
- `prompt`

For OpenPI-style evaluation, the wrapper converts DexJoCo observations into the
policy input format expected by the model server. The wrapper also handles image
resize, image dtype conversion, and camera key remapping.

The `state` field contains both proprioception and privileged environment
information. Policy inputs should use proprioception only:

- single-arm: the first 23 dimensions
- dual-arm: the first 46 dimensions

Privileged state includes task-specific environment variables such as object
poses and table parameters. Those values are useful for replay and reset
restoration, but they should not be used as policy inputs.

### Action

The OpenPI policy uses rotation vectors for end-effector orientation:

- single-arm policy action: 22 dimensions
  - `[xyz(3), rotvec(3), hand(16)]`
- dual-arm policy action: 44 dimensions
  - `[r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]`

The OpenPI environment wrapper accepts wxyz quaternion actions:

- single-arm environment action: 23 dimensions
  - `[xyz(3), quat(4), hand(16)]`
- dual-arm environment action: 46 dimensions
  - `[r_xyz(3), r_quat(4), l_xyz(3), l_quat(4), r_hand(16), l_hand(16)]`

The OpenPI client converts predicted rotation vectors to wxyz quaternions and
reorders the bimanual action before calling `env.step()`. The task's action
wrapper then splits that array into a right-arm block and a left-arm block.
Recorded demonstrations store those raw arm blocks as `[r_pose7, r_hand16,
l_pose7, l_hand16]`.

## Action Chunks

The OpenPI client waits for the first action chunk before advancing the rollout.
It aligns later chunks to their observation's control step and blends overlapping
actions. It requests a new chunk when the buffer length drops below the length
of the latest response multiplied by `--replan-ratio`, with one request in flight
at a time. Set `--replan-ratio` in `(0, 1]`. During gaps between chunks, the
simulator steps while holding the pose measured at the start of the gap.
[OpenPI evaluation loop](../dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py)

## Reusing `async_inference`

The LeRobot reference under
[`docs/reference_code/dexjoco_lerobot_client`](reference_code/dexjoco_lerobot_client)
shows how to reuse LeRobot's built-in `async_inference` support around DexJoCo.
See the official documentation at
[https://huggingface.co/docs/lerobot/async](https://huggingface.co/docs/lerobot/async).

The key idea is to separate the control loop from observation transport:

- the simulator wrapper produces observations and actions
- the `RobotClient` handles chunked action delivery and observation timing
- a background sender thread keeps observation transport off the critical path

That pattern is useful when a policy already fits the LeRobot-style async
inference API. DexJoCo also needs a rewritten robot client so the background
observation sender can run as `self._obs_sender_thread` without blocking the
control loop.

The LeRobot files are integration examples that require client-side adaptation;
they are not a packaged evaluation entry point.

## Evaluation Entry Points

Use `dexjoco-openpi-eval` for server-based evaluation.

- [OpenPI evaluation CLI](../dexjoco/dexjoco_openpi_client/cli/evaluate.py)
- [`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py`](../dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py)
- [`configs/rand_obj/*.yaml`](../configs/rand_obj)
- [`configs/rand_full/*.yaml`](../configs/rand_full)
