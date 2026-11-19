# 使用 dexmani_policy 训练和评测 DexJoCo

DP 使用 RGB，DP3 使用点云。两者都使用原生本体状态和末端 rotvec 动作。DexJoCo 提供环境、数据转换和仿真 runner，训练使用 dexmani_policy 的内置配方与下文的 DexJoCo 参数。

## 安装

在 DexJoCo 根目录执行，使用已安装 dexmani_policy 的 Conda `policy` 环境。下例假设两个仓库并列检出，`-c` 复用 dexmani_policy 的版本约束。

```bash
conda run -n policy python -m pip install \
  -c ../dexmani_policy/requirements.txt -e './dexjoco[policy]'
```

以下命令从仓库根目录通过 `scripts/run_policy.sh` 启动。脚本激活 `policy`，设置数值运行环境和 `MUJOCO_GL=egl`，再执行指定入口。

## 下载原始数据

下载固定版本的官方 v1.1 数据，只包含低维动作、元信息和初始场景：

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

```bash
task_name=water_plant
policy_method=dp3
run_dir="$PWD/outputs/train/${policy_method}_${task_name}"

bash scripts/run_policy.sh python -m dexmani_policy.train \
  --config-name "$policy_method" task_name="$task_name" \
  zarr_path="$PWD/datasets/policy/$task_name.zarr" action_key=action_rotvec \
  agent.state_key=proprio agent.state_dim=null \
  dataset.sensor_modalities.0=proprio \
  '~normalization.joint_state' +normalization.proprio=limits \
  env_runner._target_=dexjoco.policy_runner.DexJoCoRunner env_runner.env_kwargs=null \
  training.use_compile=false hydra.run.dir="$run_dir"

bash scripts/run_policy.sh python -m dexmani_policy.training.resume \
  --run "$run_dir"
```

训练 DP 时改用 `policy_method=dp`，RGB 或点云模态由所选配方提供。`zarr_path` 指向转换结果，状态与动作维度从数据推导并写入 checkpoint。以上参数加载内置网络、采样窗口和图像或点云增强，关闭 `torch.compile` 以降低首次运行成本。

`env_runner.env_kwargs=null` 清空内置配方的机器人环境参数，使用 `DexJoCoRunner` 的默认环境。续训读取 checkpoint 中保存的状态字段、runner 和其余训练配置。

## 仿真评测

```bash
bash scripts/run_policy.sh python -m dexmani_policy.evaluate \
  --checkpoint "$run_dir/checkpoints/latest.pt" --episodes 10
```

常规评测不录像；排查行为时追加 `--video`。

评测从 checkpoint 创建 `DexJoCoRunner`，复用 policy 的观测历史、动作分块和推理循环。成功、终止和步数使用原生任务状态。录像按 0.02 s 控制周期以 50 fps 保存，评测结果写入训练目录的 `eval/`。多个 checkpoint 的选择与独立测试使用 `dexmani_policy.eval_pipeline --checkpoints ...`。

默认采用 `rand_obj`：随机化物体位置和桌面高度，关闭视觉与动力学随机化。固定 seed 池为 `[10000, 20000)`，由 policy 划分互不重叠的 selection/test 集。

同一任务下，runner、环境参数和控制周期一致的 DP 与 DP3 checkpoint 可共同参与选择。省略 `env_kwargs`、设为 `null` 或空字典均使用默认环境参数。

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
