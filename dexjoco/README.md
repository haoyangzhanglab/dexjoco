# dexjoco

DexJoCo provides MuJoCo tasks, demonstration utilities and policy interfaces.

| Path | Purpose |
| --- | --- |
| `dexjoco/envs/`, `dexjoco/assets/` | Standalone v1.1 tasks and their models |
| `dexjoco/env_factory.py` | Construct a task by name |
| `dexjoco/policy_env.py` | Shared observations for Zarr export and DP/DP3 evaluation |
| `dexjoco/data/` | Demonstration download, conversion and recording |
| `dexjoco/tasks/`, `dexjoco/sim/` | Teleoperation and OpenPI task interfaces |
| `dexjoco_openpi_client/` | OpenPI WebSocket evaluation client |

## DP and DP3

Use `bash scripts/run_policy.sh ...` from the DexJoCo repository root to download
and export demonstrations in the Conda `policy` environment. Move the exported
Zarr to `dexmani_policy/robot_data/dexjoco/` for training. The policy repository owns
training and evaluation. [DP/DP3 guide](../docs/dexmani_policy.md)

The standalone task interface is Gymnasium:

```python
from contextlib import closing

from dexjoco.env_factory import make_env

with closing(make_env("water_plant", render_mode="none", action_format="rotvec")) as env:
    observation, info = env.reset(seed=0)
```

Native observations contain `agent_pos` and, when rendering, `pixels`.
`DexJoCoEnv` provides the policy fields `proprio`, `rgb` and `point_cloud`.
Both interfaces use the v1.1 right-pose, right-hand, left-pose, left-hand
ordering for bimanual control. Source details are in [UPSTREAM.md](UPSTREAM.md).

## Teleoperation and OpenPI

The [repository README](../README.md) documents the task configurations,
teleoperation recording, recorded-data replay and OpenPI evaluation commands.
Their interfaces are described in the
[custom policy guide](../docs/custom_policy_integration.md).

## License

The Gymnasium simulation foundation builds on work by
[Kevin Zakka](https://kzakka.com/).

DexJoCo source uses the repository MIT license. Bundled robot and hand assets
under `dexjoco/assets/robots/` and `dexjoco/sim/envs/xmls/` retain their own
licenses.
