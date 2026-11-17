"""Use dexmani_policy's observation history, action chunks and evaluation loop."""

import numpy as np

from dexmani_policy.env_runner.sim_runner import SimRunner

from dexjoco.common import CONTROL_TIMESTEP
from dexjoco.policy_env import DexJoCoEnv


class DexJoCoRunner(SimRunner):
    @staticmethod
    def load_seed_pool(task_name):
        # Procedural evaluation scenes; selection/test splits are owned by policy.
        return list(range(10000, 20000))

    def make_env(self, *, record_video):
        if not np.isclose(self.control_dt_s, CONTROL_TIMESTEP, rtol=1e-6, atol=1e-9):
            raise ValueError(f"DexJoCo requires control_dt_s={CONTROL_TIMESTEP}")
        return DexJoCoEnv(
            self.task_name, sensor_modalities=self.sensor_modalities,
            record_video=record_video, **self.env_kwargs,
        )
