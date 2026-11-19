"""DexJoCo environment wrapper for OpenPI inference."""

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R


class DexJoCoOpenPIEnv:
    """Adapt a DexJoCo simulation environment to the OpenPI client interface.

    The wrapper exposes processed image observations, robot state, and prompt
    fields in the format expected by the OpenPI policy server. It also converts
    policy actions from rotation-vector format to the quaternion-based action
    format used by the underlying DexJoCo environment.
    """

    def __init__(
        self,
        env_name: str,
        camera_mapping: dict[str, str],
        seed: int,
        rand_full: bool,
        randomize_dynamics: bool,
        dual_arm: bool,
        prompt: str,
        render_mode: Literal["rgb_array", "human"],
        pad_state_dim46: bool = False,
        password: list[int] | None = None,
    ):
        """Create a wrapper for one DexJoCo task environment.

        Args:
            env_name: Name used to look up the DexJoCo task config.
            camera_mapping: Mapping from OpenPI observation keys to DexJoCo
                camera keys, for example {"base": "front"}.
            seed: Environment seed.
            rand_full: Whether to randomize the environment scene.
            randomize_dynamics: Whether to randomize environment dynamics.
            dual_arm: Whether the task uses the dual-arm state/action layout.
            prompt: Language prompt passed to the policy.
            render_mode: Render mode forwarded to the DexJoCo environment.
            pad_state_dim46: Whether to pad single-arm state to 46 dimensions.
            password: List of digits representing the password for the iPad unlock task.
        """
        self.env_name = env_name
        self.camera_mapping = camera_mapping
        self.seed = seed
        self.rand_full = rand_full
        self.randomize_dynamics = randomize_dynamics
        self.dual_arm = dual_arm
        self.prompt = prompt
        self.render_mode: Literal["rgb_array", "human"] = render_mode
        self.pad_state_dim46 = pad_state_dim46
        self.password = password

        self.env = None
        self._obs: dict[str, np.ndarray] = {}
        self._stay_action: np.ndarray | None = None
        self._done = False
        self._success = False

    def start(self):
        """Instantiate the underlying DexJoCo simulation environment."""
        from dexjoco.tasks import CONFIG_MAPPING

        config = CONFIG_MAPPING[self.env_name]()

        if self.env_name == "bimanual_unlock_ipad" and self.password is not None:
            env_kwargs = {"password": self.password}
        else:
            env_kwargs = {}
            if self.password is not None:
                print(
                    f"Warning: password provided but will be ignored for env {self.env_name}"
                )

        self.env = config.get_environment(
            policy_mode=True,
            render_mode=self.render_mode,
            randomize=self.rand_full,
            seed=self.seed,
            randomize_dynamics=self.randomize_dynamics,
            **env_kwargs,
        )

    def close(self):
        """Close the underlying simulation environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self):
        """Reset the environment and cache the initial observation."""
        assert self.env is not None, "Environment not started. Call start() first."
        self._obs, _ = self.env.reset()
        self._done = False
        self._success = False
        self._stay_action = None

    def step(self, action: np.ndarray):
        """Execute one OpenPI-format action in the DexJoCo environment.

        Args:
            action: Policy action in rotation-vector format. Single-arm actions
                use [xyz(3), rotvec(3), hand(16)]. Dual-arm actions use
                [r_xyz(3), r_rotvec(3), r_hand(16),
                l_xyz(3), l_rotvec(3), l_hand(16)].
        """
        self._stay_action = None
        return self._step(self._process_action(action))

    def get_obs(self) -> dict[str, np.ndarray | str]:
        """Build the OpenPI input from the latest simulation observation."""
        from openpi_client import image_tools

        images = {
            key: image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            for key, img in self.get_raw_images().items()
        }
        state = self._obs["state"][:46 if self.dual_arm else 23]
        if self.pad_state_dim46 and not self.dual_arm:
            state = np.pad(state, (0, 23))
        return {
            **{key: images[raw_key] for key, raw_key in self.camera_mapping.items()},
            "state": state.copy(),
            "prompt": self.prompt,
        }

    def stay(self):
        """Hold the pose measured at the start of a gap between action chunks."""
        if self._stay_action is None:
            # The proprioception prefix uses the task wrapper's quaternion action order.
            self._stay_action = self._obs["state"][:46 if self.dual_arm else 23].copy()
        return self._step(self._stay_action)

    def _step(self, action: np.ndarray):
        assert self.env is not None, "Environment not started. Call start() first."
        self._obs, _, terminated, truncated, info = self.env.step(action)
        self._done = bool(terminated or truncated)
        self._success = bool(info.get("succeed", False))
        return info.get("pressed_digits")

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert OpenPI policy actions to DexJoCo environment action format."""
        if self.dual_arm:
            r_xyz = action[:3]
            r_rotvec = action[3:6]
            r_hand = action[6:22]
            l_xyz = action[22:25]
            l_rotvec = action[25:28]
            l_hand = action[28:44]

            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)

            return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])
        else:
            xyz = action[:3]
            rotvec = action[3:6]
            hand = action[6:22]

            quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)

            return np.concatenate([xyz, quat, hand])

    def get_raw_images(self) -> dict[str, np.ndarray]:
        """Return latest raw camera frames at their original resolution."""
        return {key: self._obs[key] for key in self.camera_mapping.values()}
