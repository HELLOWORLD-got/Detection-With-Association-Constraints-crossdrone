# Cross-Drone 双视图车辆复现

本目录复现 Yang 等人在 IEEE TGRS 2026 提出的多视图检测与关联框架，并运行论文表 I 的 Cross-Drone 双视图实验。模型固定为 25 个实例查询，分类头输出车辆和背景两个概率。每个实例查询在两视图间共享场景内的 Global ID，单个视图的置信度头预测该车辆在该视图是否可见。关联在同一时间戳的两个视图之间计算。

## 安装与运行

服务器上的项目目录包含 `crossdrone-data/` 时，激活已有 Python 环境后使用双卡脚本。脚本使用 GPU 0 和 1，输出到 `runs/crossdrone-resnet50/`。终端显示训练轮次、批次进度、剩余时间和训练损失。重定向到日志时，默认每 50 批及每阶段结束记录一次进度。

```bash
bash train_crossdrone.sh
bash train_crossdrone.sh --resume
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

训练默认使用 ImageNet 预训练权重。离线环境可将 `resnet50-11ad3fa6.pth` 或 `vit_b_16-c867db91.pth` 放入 `$TORCH_HOME/hub/checkpoints/`。未设置 `TORCH_HOME` 时使用 PyTorch Hub 的默认缓存路径。CPU 快速验证可使用 `--device cpu --random-init --size 64 96 --workers 0 --epochs 1 --limit-train 4 --limit-val 4` 并指定单独的输出目录。

每个训练目录保存 `config.json`、`data_audit.json`、`metrics.json`、`last.pt` 和 `best.pt`。最佳断点按验证集 MDA 选取，同分用 mAP。测试使用 `evaluate.py` 单独运行。预测 JSONL 每行对应一个时间戳，包含两个视图的原图像素框、检测分数和关联 ID。指标以 0–1 表示，乘以 100 后可与论文表格的单位对照。

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
| 式 (11)–(14) | 对象查询为 Q，实例向量与对应二维位置编码逐元素相乘并投影得到每个位置的 K，该视图全量特征为 V。每个实例仅对应自身的 Key 序列。各视图有独立的框和置信度头。 |
| 式 (15)–(25) | 按场景填充并做完整 Hungarian 匹配。分类对全部查询监督，回归对可见框在每场景归一化，置信度对所有实例和视图监督。损失与匹配系数沿用作者源码的默认数值。 |
| 式 (26)–(28)、表 I | 逐帧、逐视图对统计 MDA，再做算术平均。检测 AP 用标准 COCO IoU 0.50–0.95 和 AP50。另报告关联 precision、recall、F1 与严格 IoU > 0.60 的 UCR。 |

论文写明 AdamW、100 轮、batch size 4、学习率 `1e-4`、权重衰减 `1e-5`，并采用随机水平翻转、随机裁剪和颜色扰动。Cross-Drone 的默认输入、增强幅度和评估阈值是复现设定。输入保持原图的 16:9 比例，缩放至高 576、宽 1,024。训练时双视图共享翻转和裁剪参数，裁剪后更新框和可见性，颜色扰动分别采样。关联使用车辆概率至少 0.5、视图置信度至少 0.5 和检测 IoU 至少 0.5，MDA 的零分母视图对计入审计字段。检测 AP 按车辆概率与可见性置信度的乘积对各框排序。背景由式 (10) 的额外类别学习。

该目录对应本地 Cross-Drone 数据和表 I 中所提方法的两种骨干。按 MDA / mAP / mAP50 的顺序，论文报告 ResNet-50 为 50.61 / 28.47 / 55.23，ViT 为 51.98 / 29.21 / 56.41。当前车辆类别和序列划分采用上述复现设定。表 I 的第三方基线、其他数据集、四视图消融和实飞实验需要各自的数据及实现。

```bash
python -m unittest discover -s tests
```

[论文 DOI](https://doi.org/10.1109/TGRS.2026.3724951) · [作者公开代码](https://github.com/yyyyyxq/Instance-based-Association)
