# 基于论文独立实现的 Cross-Drone 复现

本目录从杨等人的论文图 3 和式 (1)–(28) 单独实现模型、数据、训练和评估。运行入口为 `python -m paper_reproduction`。将整个目录复制到另一个项目根目录即可使用。依赖见本目录的 `requirements.txt`。两套实现的 checkpoint 参数名和结构不同，请使用各自的评估入口。

| 论文位置 | 这里的实现 |
| --- | --- |
| 式 (5)–(8) | 共享 ImageNet 骨干，三层跨视图 CA、通道拼接降到 256、三层全局 SA。八个头，残差后做 LayerNorm。 |
| 式 (9)–(10) | 25 个全局实例查询，对加了 2D 正弦余弦位置编码的融合特征做 CA。车辆与背景通过 Softmax 学习。 |
| 式 (11)–(14)、图 3 | 每视图有独立的对象查询、CA 和预测头。同一实例的查询索引跨视图共用。Q 来源为对象查询，K 来源为全局实例向量，V 来源为单视图特征。按已确认方案，对实例向量做逐位置调制以匹配 K/V 长度。各视图通过独立 FFN 预测框和可见性。 |
| 式 (15)–(25) | 将真实实例补齐至 25；全排列 Hungarian 匹配；分类覆盖全部查询，回归只用可见框，可见性 BCE 覆盖两种负样本。 |
| 式 (26)–(28) | 逐帧逐视图对统计 MDA。UCR 要求同一查询的可见状态完全一致，且可见框均满足 IoU > 0.6。 |
| 第 V-C 节 | AdamW、全局 batch size 4、学习率 1e-4、权重衰减 1e-5、100 轮、随机翻转、裁剪和颜色扰动。 |

论文没给出单实例 Key 的空间展开、所有网络深度和损失系数，也未给出 Cross-Drone 的精确划分和检测阈值。这里使用位置调制 Key、GIQ 三层 CA、每视图 MVOQ 两层 CA、256 宽的输出 FFN 隐层、无 dropout、归一化中心宽高框，损失系数 `lambda_box/reg/conf=1`、`lambda_l1=5`、`lambda_giou=2`。系数和数据参数在命令行可设。MDA 的零分母项取 0 并计数；检测使用 COCO AP/AP50，框按车辆概率与可见性概率的乘积排序。关联默认使用车辆和可见性阈值各 0.5、匹配 IoU ≥ 0.5。检测和关联指标以 0–1 表示。训练历史也记录未过滤的最佳 IoU 均值和过线数，便于检查定位。

默认训练使用序列 1、3、4、5、6、7、8，验证用 2，测试用 0。按双视图标注 ID 合并同一物理车辆，跳过缺图的帧。只检测道路车辆。训练会排除可见车辆并集超过 25 的帧。验证和测试保留全部可用帧和标注。默认分辨率为 576×1024。数据路径示例为 `crossdrone-data/1/{1,2}/<timestamp>.png` 和 `crossdrone-data/1/1.xml`。

从仓库根目录执行：

```bash
python -m unittest paper_reproduction.tests
python -m paper_reproduction audit --data crossdrone-data

# 同图拟合。关闭数据增强，固定 8 个训练帧对并用同样的帧对评估。
CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 \
  python -u -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module paper_reproduction train --device cuda --data crossdrone-data \
  --overfit --limit-train 8 --baseline --epochs 100 --output runs/paper-fit8

# 评估产生的权重并导出预测。
CUDA_VISIBLE_DEVICES=2 python -m paper_reproduction evaluate runs/paper-fit8/last.pt \
  --data crossdrone-data --device cuda --split train \
  --output runs/paper-fit8/evaluation.json --predictions runs/paper-fit8/predictions.jsonl
```

确认同图拟合后，去掉 `--overfit --limit-train 8`，并使用新的输出目录来跑独立验证。默认使用预训练权重，优先读取仓库根目录的 `weights/resnet50-11ad3fa6.pth` 或 `weights/vit_b_16-c867db91.pth`。也可显式传入 `--weights PATH` 或选择 `--random-init`。使用 `--backbone vit_b_16` 切换骨干。续训保留原训练参数和目录并加 `--resume`，用 `--epochs` 指定新的总轮数。

本目录的实验外围覆盖 Cross-Drone。论文的其他数据集、第三方方法和实飞实验需要各自的输入数据与评估适配。表 I 的指标还需实际训练验证。
