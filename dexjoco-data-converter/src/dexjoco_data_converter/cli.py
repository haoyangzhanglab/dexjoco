import tyro


def single_lerobot_cli() -> None:
    from .to_lerobot.merge_episode_lerobot import merge_episode

    tyro.cli(merge_episode)


def single_zarr_cli() -> None:
    from .to_zarr.merge_episode_zarr import merge_episode

    tyro.cli(merge_episode)


def batch_lerobot_cli() -> None:
    from .to_lerobot.batch_merge_episode_lerobot import batch_merge_episode

    tyro.cli(batch_merge_episode)


def batch_zarr_cli() -> None:
    from .to_zarr.batch_merge_episode_zarr import batch_merge_episode

    tyro.cli(batch_merge_episode)


def merge_lerobot_cli() -> None:
    from .to_lerobot.merge_datasets_lerobot import merge_datasets

    tyro.cli(merge_datasets)


def merge_zarr_cli() -> None:
    from .to_zarr.merge_datasets_zarr import merge_datasets

    tyro.cli(merge_datasets)
