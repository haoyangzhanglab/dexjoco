from gymnasium.envs.mujoco.mujoco_rendering import (
    MujocoRenderer as GymnasiumMujocoRenderer,
)


class MujocoRenderer(GymnasiumMujocoRenderer):
    """Render each observation camera at the model's offscreen resolution."""

    def __init__(self, model, data):
        super().__init__(
            model,
            data,
            width=int(model.vis.global_.offwidth),
            height=int(model.vis.global_.offheight),
        )

    def render(self, render_mode: str | None, camera_id: int | None = None):
        if camera_id is not None:
            self.camera_id = camera_id
        return super().render(render_mode)
