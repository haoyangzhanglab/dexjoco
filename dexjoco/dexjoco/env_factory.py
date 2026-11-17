"""Construct the pinned v1.1 tasks without importing LeRobot."""

from dexjoco.envs.bimanual_assembly import BimanualAssemblyEnv
from dexjoco.envs.bimanual_hanoi import BimanualHanoiEnv
from dexjoco.envs.bimanual_microwave_cook import BimanualMicrowaveCookEnv
from dexjoco.envs.bimanual_photograph import BimanualPhotographEnv
from dexjoco.envs.bimanual_unlock_ipad import BimanualUnlockIpadEnv
from dexjoco.envs.click_mouse import ClickMouseEnv
from dexjoco.envs.fold_glasses import FoldGlassesEnv
from dexjoco.envs.hammer_nail import HammerNailEnv
from dexjoco.envs.pick_bucket import PickBucketEnv
from dexjoco.envs.pinch_tongs import PinchTongsEnv
from dexjoco.envs.water_plant import WaterPlantEnv

TaskEnv = (
    WaterPlantEnv | FoldGlassesEnv | ClickMouseEnv | PinchTongsEnv
    | PickBucketEnv | HammerNailEnv | BimanualMicrowaveCookEnv
    | BimanualUnlockIpadEnv | BimanualHanoiEnv | BimanualAssemblyEnv
    | BimanualPhotographEnv
)

TASKS = {
    env.task: env
    for env in (
        WaterPlantEnv, FoldGlassesEnv, ClickMouseEnv, PinchTongsEnv,
        PickBucketEnv, HammerNailEnv, BimanualMicrowaveCookEnv,
        BimanualUnlockIpadEnv, BimanualHanoiEnv, BimanualAssemblyEnv,
        BimanualPhotographEnv,
    )
}


def make_env(task: str, **kwargs) -> TaskEnv:
    return TASKS[task](**kwargs)
