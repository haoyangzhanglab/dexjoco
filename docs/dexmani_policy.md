# 使用 dexmani_policy 训练和评测 DexJoCo

在 DexJoCo 导出官方 v1.1 示教，再将 Zarr 移到 `dexmani_policy` 训练。评测由 `dexmani_policy` 调用已安装的 DexJoCo 环境。DP 使用 RGB，DP3 使用点云，两者都使用原生本体状态和末端 rotvec 动作。

## 安装

仅训练时，按策略仓库的说明安装 `dexmani_policy`。导出和仿真评测需要安装 DexJoCo，在 DexJoCo 根目录执行以下命令。 [dexmani_policy 安装说明](https://github.com/haoyangzhanglab/dexmani_policy#快速使用)

```bash
conda run -n policy python -m pip install -e './dexjoco[policy]'
```

如果同一环境中已安装 `dexmani_policy`，用其 `requirements.txt` 作为 `pip install -c` 的约束。记录导出时的 DexJoCo 提交（`git rev-parse HEAD`），评测安装相同提交。

## 下载原始数据

在 DexJoCo 根目录通过 `scripts/run_policy.sh` 下载固定版本的官方 v1.1 `rand_obj` 数据，包括动作 Parquet、`meta/info.json` 和 `replay_initial_values.jsonl`。

```bash
bash scripts/run_policy.sh python -m dexjoco.data.lerobot \
  --output datasets/lerobot_v11 --tasks water_plant bimanual_hanoi
```

## 数据转换

```bash
task_name=water_plant
bash scripts/run_policy.sh python -m dexjoco.data.convert_policy \
  --task "$task_name" --dataset-dir "datasets/lerobot_v11/rand_obj/$task_name" \
  --output "datasets/policy/$task_name.zarr"
```

改用 `task_name=bimanual_hanoi` 可转换双臂数据。快速验收时分别为两个任务追加 `--episodes 1`，并将 `--output` 指向 `datasets/smoke/` 下的新路径。正式转换省略 `--episodes`，处理全部示教。同一任务的 DP 和 DP3 共用转换结果。

转换恢复每条初始场景，检查 Parquet 中动作的 dtype、维度和有限性，以原始 float64 四元数动作回放，并同步生成本体状态、RGB、点云和 rotvec 监督动作。先记录 observation[t] 和 action[t]，再执行动作。按上游规则最多追加 60 步最后一个动作，遇到成功、失败或时限立即停止。

数据分批写入同一文件系统上的临时 Zarr。全部轨迹成功后才发布输出路径，失败会报告轨迹并清理临时存储。已有输出不会覆盖。

## 训练与续训

导出完成后，手动将整个 `datasets/policy/<task>.zarr` 目录移到 `dexmani_policy/robot_data/dexjoco/<task>.zarr`。两种训练配方默认从该子目录读取。其他存放位置可通过 `zarr_path` 指定。首次训练会记录数据的绝对路径，续训沿用该路径。

在 `dexmani_policy` 根目录执行以下命令。

```bash
task_name=water_plant
recipe=dp3_dexjoco
run_dir="experiments/train/${recipe}/${task_name}/run01"

bash scripts/training/train.sh "$recipe" task_name="$task_name" hydra.run.dir="$run_dir"

conda run --no-capture-output -n policy python -m dexmani_policy.training.resume --run "$run_dir"
```

训练 DP 时使用 `recipe=dp_dexjoco`。通过 `task_name` 选择任务，两种配方都从 Zarr 推导状态与动作维度，并把评测使用的 runner 路径写入 checkpoint。训练和续训只需要策略依赖和 Zarr。

## 仿真评测

在 `dexmani_policy` 根目录执行以下命令。

```bash
MUJOCO_GL=egl bash scripts/eval/evaluate.sh \
  --checkpoint "$run_dir/checkpoints/latest.pt" --episodes 10
```

追加 `--video` 可保存录像。

评测从 checkpoint 创建 `DexJoCoRunner`，复用策略仓库的观测历史、动作分块和推理循环。结果写入训练目录下的 `eval/`。`result_details.json` 记录控制周期和相对录像路径 `videos/<task>/episode_<seed>.mp4`。启用录像时，从 224×224 的 ego 相机在每个控制步后取一帧，按原生控制周期以 50 fps 编码。

`total_steps` 记录实际执行的控制步数。DexJoCo 的 `steps` 记录原生任务确认成功的步数，因此成功时等于 `total_steps`。`dexmani_sim` 的 `steps` 记录首次满足原始成功条件的步数，确认成功还可能需要持续多个控制步。

多个 checkpoint 的选择与独立测试使用 `MUJOCO_GL=egl bash scripts/eval/eval_pipeline.sh --checkpoints ...`。

默认采用 `rand_obj`，随机化物体位置和桌面高度，关闭视觉与动力学随机化。固定 seed 池为 `[10000, 20000)`，由策略仓库划分互不重叠的 selection/test 集。

同一任务下，runner、环境参数和控制周期一致的 DP 与 DP3 checkpoint 可共同参与选择。

## 数据约定

| 字段 | 单臂 | 双臂 | 含义 |
| --- | --- | --- | --- |
| `proprio` | 23 | 46 | TCP 位姿与手指关节，float32，无任务特权状态 |
| `action_rotvec` | 22 | 44 | 每臂包含绝对 TCP xyz、旋转向量和 16 个手指目标，float32 |
| `rgb` | 224×224×3 | 224×224×3 | 原生 ego 相机图像，uint8 |
| `point_cloud` | 4096×6 | 4096×6 | 世界系 XYZ（米）和 RGB [0,1]，float32 |

双臂状态和动作按**右臂位姿、右手、左臂位姿、左手**排列，四元数为 wxyz。右手顺序为 ff、mf、rf、th，左手为 rf、mf、ff、th。rotvec 转换使用上游 `quaternion_to_rotvec`。

数据转换和在线评测共用 `DexJoCoEnv`。ego 对应单臂 `front`、双臂 `back`；RGB 与深度来自同一个原生 renderer，保留喷水和相机屏幕等效果。补充深度时不再次推进视觉更新。

MuJoCo 输出米制深度；按像素中心、垂直 FOV 和相机位姿反投影到世界系。相机坐标为 x 向右、y 向上、−z 向前。过滤无效及远裁剪面像素后均匀选取 4096 点，不足时重复；DP3 再通过 FPS 选取 1024 点。

转换和评测配置均为 224×224、offsamples=0。Zarr 属性记录数据来源、`simulator_revision`、实际观测参数、动作空间 `eef_rotvec` 和控制周期 `dt=0.02`；物理周期为 0.002 s。

## 来源

仿真来源和复制边界见 [UPSTREAM.md](../dexjoco/UPSTREAM.md)。官方数据为 `DexJoCo/DexJoCo-Datasets-v1.1`，revision 固定为 `077bbed4a3c9807ae88dad06842b7989533ab2bc`。
