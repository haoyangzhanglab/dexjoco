# 使用 dexmani_policy 训练和评测 DexJoCo

DP 使用 RGB，DP3 使用点云；两者都使用原生本体状态和末端 rotvec 动作。DexJoCo 提供环境、数据转换和配方，dexmani_policy 提供训练、续训与评测。

## 安装

在 DexJoCo 根目录执行，使用已安装 dexmani_policy 的 Conda `policy` 环境：

```bash
conda run -n policy python -m pip install -c requirements-policy.txt -e './dexjoco[policy]'
```

所有命令从仓库根目录通过 `scripts/run_policy.sh` 启动。脚本激活 `policy`，配置数值运行环境、`MUJOCO_GL=egl` 和 `DEXJOCO_ROOT`，并通过 `exec` 传递终止信号。训练和渲染需要访问 GPU 驱动。

## 下载原始数据

下载固定版本的官方 v1.1 数据，只包含低维动作、元信息和初始场景：

```bash
bash scripts/run_policy.sh python -m dexjoco.data.lerobot \
  --output datasets/lerobot_v11 --tasks water_plant bimanual_hanoi
```

## 快速验收

在仓库根目录执行；将 `task_name` 换成 `bimanual_hanoi` 验证双臂，将 `policy_method` 换成 `dp` 验证 RGB 策略。使用新的输出目录，已有数据和训练产物不会覆盖。

```bash
task_name=water_plant
policy_method=dp3
run_dir="$PWD/outputs/smoke/${policy_method}_${task_name}"

bash scripts/run_policy.sh python -m dexjoco.data.convert_policy \
  --task "$task_name" --dataset-dir "datasets/lerobot_v11/rand_obj/$task_name" \
  --output "datasets/smoke/$task_name.zarr" --episodes 5

bash scripts/run_policy.sh python -m dexmani_policy.test_tools.smoke_test "${policy_method}_dexjoco" \
  --config-dir configs/dexmani_policy --output-dir "$run_dir" \
  --override "task_name=$task_name" \
  --override "zarr_path=$PWD/datasets/smoke/$task_name.zarr" \
  --override dataloader.batch_size=8 --override dataloader.num_workers=0

bash scripts/run_policy.sh python -m dexmani_policy.evaluate \
  --checkpoint "$run_dir/checkpoints/latest.pt" --episodes 1
```

同一任务的 DP、DP3 共用转换结果。smoke 默认训练 4 步、续训 2 步，检查模型、EMA、优化器和调度器恢复；步数由 `--max-updates` 和 `--resume-updates` 控制。相对路径以启动目录为准，省略 `--output-dir` 时自动清理临时产物。

短验收仅检查流程连通，不评价策略性能。正式配方默认读取 `datasets/policy/`，验收数据放在 `datasets/smoke/`。

## 正式数据转换

```bash
task_name=water_plant
bash scripts/run_policy.sh python -m dexjoco.data.convert_policy \
  --task "$task_name" --dataset-dir "datasets/lerobot_v11/rand_obj/$task_name" \
  --output "datasets/policy/$task_name.zarr"
```

改用 `task_name=bimanual_hanoi` 可转换双臂数据。不指定 `--episodes` 时转换全部示教。少量示教验收使用 `datasets/smoke/`。

转换恢复每条初始场景，检查 Parquet 中动作的实际 dtype，以原始 float64 四元数动作回放，并同步生成本体状态、RGB、点云和 rotvec 监督动作。先记录 observation[t] 和 action[t]，再执行动作；按上游规则最多追加 60 步最后一个动作，遇到成功、失败或时限立即停止。

数据每 64 帧写入 Zarr。任何轨迹失败都会报错；全部成功后才将 `.partial` 目录改为最终名称。已有输出不会覆盖。

## 训练与续训

```bash
task_name=water_plant
policy_method=dp3  # dp 或 dp3
run_dir="$PWD/outputs/train/${policy_method}_${task_name}"

bash scripts/run_policy.sh python -m dexmani_policy.train \
  --config-dir "$PWD/configs/dexmani_policy" --config-name "${policy_method}_dexjoco" \
  task_name="$task_name" hydra.run.dir="$run_dir"

bash scripts/run_policy.sh python -m dexmani_policy.training.common.resume \
  --run "$run_dir"
```

默认读取 `datasets/policy/${task_name}.zarr`，也可指定 `zarr_path=/absolute/path/to/data.zarr`。状态与动作维度从数据推导并写入 checkpoint。配方保留原网络、采样窗口和图像/点云增强，关闭 `torch.compile` 以降低首次运行成本。

续训使用 checkpoint 中保存的配置。`agent.state_key` 指定本体状态字段，`env_runner._target_` 指定运行器，二者随配置保存到 checkpoint。

## 仿真评测

```bash
bash scripts/run_policy.sh python -m dexmani_policy.evaluate \
  --checkpoint "$run_dir/checkpoints/latest.pt" --episodes 10
```

常规评测不录像；排查行为时追加 `--video`。

评测从 checkpoint 创建 `DexJoCoRunner`，复用 policy 的观测历史、动作分块和推理循环。成功、终止和步数使用原生任务状态；控制频率为 50 Hz，wrapper 的 `metadata["render_fps"]` 与 `video_fps` 均为 50。结果保存到训练目录的 `eval/`。多个 checkpoint 的选择与独立测试使用 `dexmani_policy.eval_pipeline --checkpoints ...`。

默认采用 `rand_obj`：随机化物体位置和桌面高度，关闭视觉与动力学随机化。固定 seed 池为 `[10000, 20000)`，由 policy 划分互不重叠的 selection/test 集。

DP 与 DP3 配方显式记录相同的环境参数。同一任务下，两者可共同参与 checkpoint 选择；自定义配方也须保持相同环境参数和控制周期。

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

## 来源与检查

仿真来源和复制边界见 [UPSTREAM.md](../dexjoco/UPSTREAM.md)。官方数据为 `DexJoCo/DexJoCo-Datasets-v1.1`，revision 固定为 `077bbed4a3c9807ae88dad06842b7989533ab2bc`。

配方、示教精度与时序、相机几何和任务接口使用统一测试入口：

```bash
bash scripts/run_policy.sh python -m pytest tests/test_policy_env.py -q
```
