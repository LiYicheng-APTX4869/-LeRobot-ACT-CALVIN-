# 基于 LeRobot ACT 的 CALVIN 跨环境泛化实验

本仓库是 HW3 具身智能任务的实验代码仓库，目标是在 CALVIN 数据集上使用
LeRobot 框架训练和评估 ACT（Action Chunking Transformer）动作策略，并比较
单环境训练与多环境联合训练在未见环境上的 zero-shot 泛化能力。

实验包含三部分：

1. 仅使用 CALVIN 环境 A 训练 A-only ACT 策略；
2. 使用环境 A、B、C 的混合数据，在相同网络结构和超参数下训练 ABC-joint ACT 策略；
3. 在完全未见过的环境 D 上进行 zero-shot 离线动作误差评估。

最终 splitD 评估结果如下：

| 模型 | Action L1 | Action MSE | Action L2 |
|---|---:|---:|---:|
| A-only | 0.149435 | 0.064812 | 0.552181 |
| ABC-joint | 0.125359 | 0.054153 | 0.471198 |
| 相对降低 | 16.11% | 16.45% | 14.67% |

结果表明，多环境联合训练能够显著降低未见环境 D 上的动作误差，提升 ACT 在视觉分布偏移下的泛化能力。

## 目录结构

```text
configs/
  act_calvin_common.env        # 公共路径、训练超参数和脚本配置。
scripts/
  00_setup_env.sh              # 安装实验依赖。
  01_download_data.sh          # 下载 CALVIN LeRobot 数据。
  01_convert_data_v30.sh       # 将旧版 LeRobot 数据转换为 v30 格式。
  02_train_A_only.sh           # 仅使用 splitA 训练 ACT。
  03_train_ABC_joint.sh        # 合并 splitA/B/C 并训练联合模型。
  04_eval_on_D.sh              # 在 splitD 上评估单个 checkpoint。
tools/
  *.py                         # 数据检查、合并、转换、评估和绘图辅助脚本。
```

## 环境配置

推荐环境：

- Linux 服务器；
- NVIDIA GPU；
- Conda 或 Miniconda；
- Python 3.10；
- 足够的磁盘空间用于 CALVIN 数据、v30 转换数据、Hugging Face cache 和 checkpoint。

创建并激活 Conda 环境：

```bash
conda create -n lerobot-act python=3.10 -y
conda activate lerobot-act
```

方式一：使用安装脚本安装依赖：

```bash
bash scripts/00_setup_env.sh
```

方式二：使用 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate lerobot-act
```

方式三：使用 `requirements.txt` 安装 pip 依赖：

```bash
python -m pip install -r requirements.txt
```

如果服务器用户目录空间较小，建议将临时目录和缓存放到大盘：

```bash
TMP_ROOT=/HDD_DISK/users/$USER/tmp \
CACHE_ROOT=/HDD_DISK/users/$USER/hw3_cache \
bash scripts/00_setup_env.sh
```

## 数据准备

下载 CALVIN LeRobot 数据：

```bash
bash scripts/01_download_data.sh
```

如果 Hugging Face 访问较慢或无法连接，可以使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/01_download_data.sh
```

将下载得到的数据转换为当前 LeRobot 需要的 v30 格式：

```bash
FORCE_RECONVERT=1 bash scripts/01_convert_data_v30.sh splitA splitB splitC splitD
```

转换完成后，预期得到：

```text
data/calvin-lerobot/splitA_v30
data/calvin-lerobot/splitB_v30
data/calvin-lerobot/splitC_v30
data/calvin-lerobot/splitD_v30
```

可以使用以下命令检查转换后的数据：

```bash
HF_DATASETS_CACHE=outputs/cache/hf_datasets \
python tools/inspect_v30_dataset.py \
  --dataset-root data/calvin-lerobot/splitA_v30 \
  --repo-id local/calvin_splitA_v30
```

## 模型训练

训练 A-only 基础模型：

```bash
FORCE_RESTART=1 \
TRAIN_STEPS=100000 \
SAVE_FREQ=10000 \
EVAL_FREQ=10000 \
LOG_FREQ=100 \
bash scripts/02_train_A_only.sh
```

训练 ABC-joint 联合模型：

```bash
FORCE_RESTART=1 \
TRAIN_STEPS=100000 \
SAVE_FREQ=10000 \
EVAL_FREQ=10000 \
LOG_FREQ=100 \
bash scripts/03_train_ABC_joint.sh
```

常用覆盖参数示例：

```bash
GPU_IDS=0,1 BATCH_SIZE=64 NUM_WORKERS=8 bash scripts/02_train_A_only.sh
GPU_IDS=0,1 BATCH_SIZE=64 NUM_WORKERS=8 bash scripts/03_train_ABC_joint.sh
```

最终 checkpoint 路径为：

```text
outputs/act_calvin_A_only/checkpoints/last/pretrained_model
outputs/act_calvin_ABC_joint/checkpoints/last/pretrained_model
```

## 在未见环境 D 上测试

本实验采用 splitD 离线动作误差作为 zero-shot 泛化指标。评估时，将策略在 splitD 观测上预测的动作与数据集中专家动作进行比较，报告 Action L1、Action MSE 和 Action L2。

快速小规模测试：

```bash
GPU_IDS=0 EVAL_EPISODES=20 EVAL_MAX_BATCHES=20 \
bash scripts/04_eval_on_D.sh act_calvin_A_only
```

完整分块评估 A-only 模型：

```bash
GPU_IDS=0 CHUNK_EPISODES=200 START_EPISODE=0 MAX_EPISODES=2600 \
bash scripts/07_eval_D_offline_chunks.sh act_calvin_A_only
```

完整分块评估 ABC-joint 模型：

```bash
GPU_IDS=0 CHUNK_EPISODES=200 START_EPISODE=0 MAX_EPISODES=2600 \
bash scripts/07_eval_D_offline_chunks.sh act_calvin_ABC_joint
```

聚合后的评估结果保存在：

```text
outputs/act_calvin_A_only/eval_splitD/offline_action_error_aggregated.json
outputs/act_calvin_ABC_joint/eval_splitD/offline_action_error_aggregated.json
```

## 复现实验注意事项

- splitD 只用于 zero-shot 测试，不能用于训练、验证、超参数搜索或 early stopping。
- A-only 和 ABC-joint 必须使用相同 ACT 网络结构与超参数。
- 本报告中的完整 splitD 评估使用相同 episode 范围、相同 batch 数和相同帧数。
- 如果完整 splitD 评估时出现 `No space left on device`，请使用 `scripts/07_eval_D_offline_chunks.sh`，该脚本会在每个 chunk 之间清理缓存。
