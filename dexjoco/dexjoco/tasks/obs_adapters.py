from collections import OrderedDict
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space


class DexjocoObsAdapter(gym.ObservationWrapper):
    """Adapt raw task observations into the flattened Dexjoco collection format."""

    def __init__(self, env, proprio_keys: list[str]):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        raw_spaces = cast(gym.spaces.Dict, env.observation_space).spaces
        state_spaces = cast(gym.spaces.Dict, raw_spaces["state"]).spaces
        images = raw_spaces.get("images")
        camera_spaces = cast(gym.spaces.Dict, images).spaces if images is not None else {}

        ordered_spaces = OrderedDict(
            (key, state_spaces[key]) for key in self.proprio_keys
        )
        flat_space = cast(gym.spaces.Box, flatten_space(gym.spaces.Dict(ordered_spaces)))
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(flat_space.low, flat_space.high, dtype=np.float64),
                **camera_spaces,
            }
        )

    def observation(self, obs):
        proprio_values = [np.asarray(obs["state"][key]).ravel() for key in self.proprio_keys]
        flat_state = np.concatenate(proprio_values, axis=0).astype(np.float64, copy=False)
        return {"state": flat_state, **obs.get("images", {})}
