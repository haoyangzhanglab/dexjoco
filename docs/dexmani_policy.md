# DexJoCo 示教导出、训练与评测

DexJoCo 提供任务环境、示教导出和共享观测。两个仓库以 Zarr 交接数据。训练只依赖 `dexmani_policy` 和 Zarr，导出与评测使用安装了 `dexjoco[policy]` 的 `policy` 环境。

## 1. 在 DexJoCo 导出示教

`dexjoco.data.lerobot` 下载官方 v1.1 `rand_obj` 示教，`dexjoco.data.convert_policy` 回放原始动作并导出 Zarr。导出和评测共用 `DexJoCoEnv`。RGB 取自原生 ego 相机（单臂 `front`，双臂 `back`），点云由同一相机的深度反投影到世界坐标，单位为米。每帧观测在对应动作执行前采集。

以 `water_plant` 为例，在 DexJoCo 根目录执行。

```bash
bash scripts/run_policy.sh python -m dexjoco.data.lerobot \
  --output datasets/lerobot_v11 --tasks water_plant
bash scripts/run_policy.sh python -m dexjoco.data.convert_policy \
  --task water_plant --dataset-dir datasets/lerobot_v11/rand_obj/water_plant \
  --output datasets/policy/water_plant.zarr
```

将整个 Zarr 目录手动移到 `dexmani_policy/robot_data/dexjoco/water_plant.zarr`。

## 2. 在 dexmani_policy 训练策略

自定义配方的 `defaults` 先引用 `dp`（RGB）或 `dp3`（点云），再引用共用的 `dexjoco` 配置。两种策略共用同一份 Zarr，以 `proprio` 为本体状态，以 `action_rotvec` 为监督目标。

在 `dexmani_policy` 根目录运行，将 `your_recipe` 替换为配方名。

```bash
bash scripts/training/train.sh your_recipe task_name=water_plant \
  hydra.run.dir=experiments/train/water_plant/run01
```

## 3. 在 dexmani_policy 评测策略

评测从 checkpoint 恢复策略。`DexJoCoRunner` 调用 `DexJoCoEnv`，复用 `SimRunner` 的闭环评测流程，向环境发送原生绝对旋转向量动作。在 `dexmani_policy` 根目录运行。

```bash
MUJOCO_GL=egl bash scripts/eval/evaluate.sh \
  --checkpoint experiments/train/water_plant/run01/checkpoints/latest.pt --episodes 10
```

回合结果写入 `eval/` 下的 `result_details.json`，追加 `--video` 可同时保存录像。

[策略安装说明](https://github.com/haoyangzhanglab/dexmani_policy#快速使用) · [DexJoCo 仿真来源](../dexjoco/UPSTREAM.md)
