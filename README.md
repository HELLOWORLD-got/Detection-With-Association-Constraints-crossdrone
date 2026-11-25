# Cross-Drone 双视图车辆复现

本目录复现 Yang 等人在 IEEE TGRS 2026 提出的多视图检测与关联框架，并运行论文表 I 的 Cross-Drone 双视图实验。模型固定为 25 个实例查询，分类头输出车辆和背景两个概率。每个实例查询在两视图间共享场景内的 Global ID，单个视图的置信度头预测该车辆在该视图是否可见。关联在同一时间戳的两个视图之间计算。

## 安装与运行

服务器上的项目目录包含 `crossdrone-data/` 时，激活已有 Python 环境后使用双卡脚本。脚本默认使用 GPU 0 和 1，可通过 `CUDA_VISIBLE_DEVICES` 选择其他两张卡。默认输出到 `runs/crossdrone-resnet50/`。Rich 进度条区分训练和验证，并显示轮次、批次、已用时间、预计剩余时间和训练损失。直接在终端运行时实时刷新，重定向到日志时默认每 50 批及每阶段结束记录一次进度。

```bash
bash train_crossdrone.sh
bash train_crossdrone.sh --resume
```

CUDA 双卡的 NCCL 同步遇到问题时，可以用同一启动脚本尝试 Gloo 后端。先用少量数据验证当前 PyTorch 构建的 CUDA 通信支持。

```bash
GLOO_SOCKET_IFNAME=lo bash train_crossdrone.sh --distributed-backend gloo \
  --epochs 1 --limit-train 8 --limit-val 8 --output runs/startup-check-gloo
```

在项目根目录执行。Python 3.10 或以上可用。CUDA 环境使用与你的驱动匹配的 PyTorch 和 torchvision。

```bash
python -m pip install -r requirements.txt
python train.py --audit-only

# 论文训练参数，双 GPU，两个进程合计 batch size 为 4
torchrun --standalone --nproc-per-node=2 train.py --device cuda --output runs/crossdrone-resnet50

# 单 GPU 使用同一全局 batch size
python train.py --device cuda --output runs/crossdrone-resnet50

# 表 I 的另一种骨干
torchrun --standalone --nproc-per-node=2 train.py --device cuda \
  --backbone vit_b_16 --output runs/crossdrone-vit

# 使用启动该训练时的相同参数，且 --epochs 大于已保存的轮数
torchrun --standalone --nproc-per-node=2 train.py --device cuda \
  --output runs/crossdrone-resnet50 --resume

python evaluate.py runs/crossdrone-resnet50/best.pt --split test --device cuda \
  --output runs/crossdrone-resnet50/test.json \
  --predictions runs/crossdrone-resnet50/test-predictions.jsonl
```

训练默认使用 ImageNet 预训练权重。手动下载的文件可直接放在项目根目录的 `weights/` 内，默认 ResNet-50 使用 `resnet50-11ad3fa6.pth`，ViT-B/16 使用 `vit_b_16-c867db91.pth`。项目目录缺少对应文件时，TorchVision 会检查 PyTorch Hub 缓存，并在需要时下载。CPU 快速验证可使用 `--device cpu --random-init --size 64 96 --workers 0 --epochs 1 --limit-train 4 --limit-val 4` 并指定单独的输出目录。

每个训练目录保存 `config.json`、`data_audit.json`、`metrics.json`、`last.pt` 和 `best.pt`。最佳断点按验证集 MDA 选取，同分用 mAP。测试使用 `evaluate.py` 单独运行。预测 JSONL 每行对应一个时间戳，包含两个视图的原图像素框、检测分数和关联 ID。指标以 0–1 表示，乘以 100 后可与论文表格的单位对照。

### 小样本性能检查

先在固定的 128 个验证图像对上测初始模型，再用 384 个训练对跑 12 轮。两次评估使用相同阈值、分辨率和验证样本。若服务器需要关闭 NCCL P2P，使用下列命令。

```bash
NCCL_P2P_DISABLE=1 bash train_crossdrone.sh --validate-before-training \
  --epochs 12 --limit-train 384 --limit-val 128 \
  --output runs/learning-check
```

初始验证写入 `baseline.json`，各轮验证写入 `metrics.json`。对照同一验证集上的 MDA、mAP 和 mAP50。小样本取自各划分的前若干对，受相邻帧和场景分布影响。

## 数据和划分

- 路径使用 `数据集/<序列>/<1|2>/<timestamp>.png` 和 `数据集/<序列>/<序列>.xml`。精确匹配时间戳字符串。两视图按原始车辆 ID 取并集，`outside=0` 且裁剪到图像后面积为正的框记为可见。
- 车辆前景包括 `CAR_`、轿车、SUV、厢式车、露营车、卡车和道路挂车模型。火车和叉车计为背景。审计文件记录排除的 XML ID、缺失图像和零面积框。
- 当前 XML 共 7,600 对。序列 3 缺第一视图 943 张，序列 4 缺第一视图 201 张，共 6,456 对可用。默认按目录分配完整序列。论文的 80/10/10 航次划分写在自建数据集段落，Cross-Drone 段落未给出官方划分。当前配置在完整双视图对上约为 79/10/11。

| 集合 | 序列 | 完整双视图对 | 超过 25 辆车 | 参与训练或评估 |
| --- | --- | ---: | ---: | ---: |
| 训练 | 1、3、4、5、6、7、8 | 5,100 | 1,630 | 3,470 |
| 验证 | 2 | 641 | 458 | 641 |
| 测试 | 0 | 715 | 555 | 715 |

训练按增强前的可见车辆并集排除超出容量的整帧，对保留帧应用式 (15)–(25)。验证和测试使用全部完整图像对及全部车辆标注。单帧最多输出 25 个关联实例，因此高密度帧会体现模型的召回上限。修复缺图后建议重新运行数据审计。评估结果会标记当前数据索引是否与训练断点一致。

## 公式和复现设定

| 论文位置 | 实现 |
| --- | --- |
| 式 (1)–(8) | 共享 ResNet-50 或 ViT-B/16，MVFF 先三层 CA，再通道拼接和三层 SA。沿用源码的 `1×1` 拼接降维。 |
| 图 3、式 (9)–(10) | GIQ 对带固定二维正弦余弦位置编码的全局特征做交叉注意力，输出实例特征和车辆/背景概率。 |
| 式 (11)–(14) | 对象查询为 Q，实例向量与对应二维位置编码逐元素相乘并投影得到每个位置的 K，该视图全量特征为 V。每个实例仅对应自身的 Key 序列。图 3 的每视图分支各有一套 CA Blocks、框 FFN 和置信度 FFN。论文未明说 CA 是否共享参数，本实现按图将每个视图的 CA 作为独立模块。 |
| 式 (15)–(25) | 按场景填充并做完整 Hungarian 匹配。分类对全部查询监督，回归对可见框在每场景归一化，置信度对所有实例和视图监督。损失与匹配系数沿用作者源码的默认数值。 |
| 式 (26)–(28)、表 I | 逐帧、逐视图对统计 MDA，再做算术平均。检测 AP 用标准 COCO IoU 0.50–0.95 和 AP50。另报告关联 precision、recall、F1 与严格 IoU > 0.60 的 UCR。 |

论文写明 AdamW、100 轮、batch size 4、学习率 `1e-4`、权重衰减 `1e-5`，并采用随机水平翻转、随机裁剪和颜色扰动。Cross-Drone 的默认输入、增强幅度和评估阈值是复现设定。输入保持原图的 16:9 比例，缩放至高 576、宽 1,024。训练时双视图共享翻转和裁剪参数，裁剪后更新框和可见性，颜色扰动分别采样。关联使用车辆概率至少 0.5、视图置信度至少 0.5 和检测 IoU 至少 0.5，MDA 按式 (26) 除以全部帧与视图对的数量。论文未定义分母为零的单项分数，本实现将该项记为 0，同时记录其数量。检测 AP 按车辆概率与可见性置信度的乘积对各框排序。背景由式 (10) 的额外类别学习。

这次按图 3 拆分了原来跨视图共享的 MVOQ CA 栈。早期权重仍可由 `evaluate.py` 按旧架构读取，报告用 `mvoq_attention_shared` 标识。新架构应使用新的输出目录从 ImageNet 初始化开始训练，旧实验的优化器状态不能用于新架构。式 (11) 规定了 Q/K/V 的来源，未说明单个 instance Key 如何对齐空间 Value；位置调制是本项目采用的补全。论文也未给出 GIQ/MVOQ 深度、Cross-Drone 的预处理与划分或全部损失系数，因此这些设置保留为复现约定。

该目录对应本地 Cross-Drone 数据和表 I 中所提方法的两种骨干。按 MDA / mAP / mAP50 的顺序，论文报告 ResNet-50 为 50.61 / 28.47 / 55.23，ViT 为 51.98 / 29.21 / 56.41。当前车辆类别和序列划分采用上述复现设定。表 I 的第三方基线、其他数据集、四视图消融和实飞实验需要各自的数据及实现。

```bash
python -m unittest discover -s tests
```

[论文 DOI](https://doi.org/10.1109/TGRS.2026.3724951) · [作者公开代码](https://github.com/yyyyyxq/Instance-based-Association)

## 零指标排查（修复后）

本次修复：置信度使用原始 logits 计算稳定 BCE，避免 clamp 截断错误高置信度预测的梯度；冻结骨干时同时固定 BatchNorm 状态；微小框 GIoU 避免固定面积下限产生虚假误差；Hungarian 非有限代价立即报错，反向传播后检查梯度再更新参数。上述数值修复支持旧权重评估。图 3 的 MVOQ 架构差异另见上一节。这些修复不等于已经证明服务器上的零 AP 根因，也不保证恢复旧断点后立即收敛。

`evaluate.py --split val` 默认继承 checkpoint 的 `limit_val`；`--split train` 先按 N=25 排除超容量帧，再继承 `limit_train`，关闭数据增强。`--limit N` 显式覆盖；`--full` 取消限额。独立评估记录 checkpoint epoch 和数据索引变化。过拟合诊断的 val 始终来自该实验的训练子集，不能当作独立验证成绩。

同步代码到服务器后，在项目根目录和已有环境中执行：

```bash
# 1. 回归测试；依赖沿用 requirements.txt。
python -m unittest discover -s tests

# 2. 诊断已有的小样本实验，使用最后一轮权重。
# 全部指标为零时 best.pt 可能仍然是第 1 轮。
CUDA_VISIBLE_DEVICES=2,3 bash debug_crossdrone.sh diagnose \
  runs/convergence-gpu23/last.pt runs/diagnostics-existing

# 3. 新开实验：8 个图像对，关闭增强，同图评估，保留默认分辨率、学习率、N=25。
# 两张 GPU；默认关闭 NCCL P2P。200 轮是诊断预算，不是论文正式训练配置。
CUDA_VISIBLE_DEVICES=2,3 bash debug_crossdrone.sh overfit runs/overfit-8-fixed
```

`diagnose` 使用第一张可见卡单进程运行；训练和验证使用各自保存的样本限额，输出 `train.json`、`val.json` 及各自日志。`overfit` 使用双卡，输出 `baseline.json`、`metrics.json`、`last.pt`、`best.pt`、`train.log`，完成后自动输出诊断。请使用新输出目录，并确认 GPU 没有被其他实验占用。中途报错时保留 `train.log`，不自动重试掩盖问题。

JSON 的 `diagnostics.summary` 中：

- `raw_oracle_coverage_iou50`：忽略所有置信度，GT 是否至少被一个预测框以 IoU≥0.5 覆盖。它允许同一个预测覆盖多个 GT，**不是正式召回率**。
- `kept_oracle_coverage_iou50`：经过车辆/可见性阈值后剩余框的覆盖率。raw 高、kept 低，优先检查分类和可见性；两者都低，优先检查定位。
- `raw_best_iou`、`pred_width_px` / `gt_width_px`、`pred_height_px` / `gt_height_px`：定位和尺寸分布。像素单位使用原图尺寸。
- `foreground_probability`、`visibility_probability`、`confidence_abs_logit`：置信度分布；logit 绝对值很大时检查是否饱和。
- `center_error_px`、`size_error_px`：选择每个 GT 的最大 IoU 框计算平均绝对坐标误差；所有 IoU 为零时选择任意并列框，不能单独用这一项下结论。

`diagnostics.frames` 记录逐帧、逐视图结果和样本标识，便于找出异常标注或场景。过拟合实验还在每轮 `train.last_step_gradient_norms` 保存最后一步各模块梯度范数；`null` 表示没有梯度，0 表示该步范数为零，连续异常才值得进一步排查。

判断顺序：先确认数据索引、样本数和 checkpoint epoch；再看同图训练的框覆盖率和 AP 是否上升。如果同图可以拟合而独立验证差，检查场景分布及小样本代表性。如果同图长期无法拟合且原始覆盖率接近零，继续检查框预测与 MVOQ 注意力的学习能力，并做独立消融。这里仍使用已确认的位置调制 Key 方案，不因指标低而自动改变论文公式或扩充查询数。正式复现时不要启用 `--overfit`。

### MVOQ 局部特征对照

定位实验可以在相同样本、种子和训练轮数下，比较默认的位置调制 Key 与加入局部外观的 Key。两者都使用每视图独立的 CA。后者通过 `--mvoq-content-key` 启用，把当前视图的空间特征加到位置编码后，再用对应的全局实例特征调制。参数形状相同，属于对式 (11) 未公开细节的消融。默认训练使用位置调制 Key。

```bash
export CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1
bash train_crossdrone.sh --overfit --limit-train 8 --epochs 100 --workers 2 \
  --validate-before-training --output runs/overfit-8-position-key
bash train_crossdrone.sh --overfit --limit-train 8 --epochs 100 --workers 2 \
  --validate-before-training --mvoq-content-key --output runs/overfit-8-content-key
```

同图实验的初始和逐轮评估会记录 `raw_best_iou` 与 `raw_oracle_coverage_iou50`。优先按相同训练步数比较这些量和 `mAP50`，再看损失。评估 checkpoint 时会自动使用保存的 Key 设置。
