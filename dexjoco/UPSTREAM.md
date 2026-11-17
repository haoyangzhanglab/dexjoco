# v1.1 simulation core

Source: https://github.com/brave-eai/dexjoco/tree/12ea8ffea2671072f4630a0c83d9233f308a59d4

`dexjoco/actions.py`, `common.py`, `controllers.py`, `envs/` and `assets/`
are copied unchanged from that revision's `src/dexjoco/`. Asset licenses remain
beside the models. The root repository license covers DexJoCo source code.

The local `env_factory.py` exposes a single Gymnasium environment without the
upstream LeRobot configuration and vector-environment dependencies. The
teleoperation and OpenPI interfaces use `sim/` and `tasks/`.

The Conda `policy` environment uses dm-robotics-transformations 0.9.0 with
NumPy 1.26.4. Data conversion checks native replay success before publishing
the training dataset.
