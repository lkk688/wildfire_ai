# 确认性复现计划（预注册）

写定日期：2026-10-20，**在任何模型跑确认集之前提交**。提交后，下列内容不再修改：假设、数据、模型、提示词、解析规则和分析代码。如有偏离，在结果文档中逐条列出。

## 1. 目的

试点集和配对集都来自 2024–2025 年的测试事件，相关结论是在同一批数据上反复查看后得出的，存在多重比较和"分叉路径"的风险。本计划在**从未被任何 VLM 跑过的 19 场 val 事件**（2019–2023 年）上，复现二级结论中最关键的几条。

## 2. 数据（已构建并冻结）

| 文件 | 内容 | SHA256 |
|---|---|---|
| `output/splits/confirm_pairs.jsonl` | 289 对邻居，同一场 val 火、相距 15–100 m，其中一栋被毁一栋存活；来自 9 场火，Glass 占 175 对（61%）；排除与其他划分相距 100 m 以内的建筑 | `8021687b0a523ff1b9ac6a08b765cff4cbfd00301e49cdf34ab3fcee97afabfa` |
| `output/splits/confirm_direct.jsonl` | 220 栋，取自 6 场 val 火，每场火每类最多 20 栋（平衡）；与配对集互不重叠 | `ef970e0bd67f34085e7b209ee0f96a1a9b3795bd70874551eb18323f7c29c287` |

构建脚本：`scripts/data/build_pair_set.py --split val --min-per-class 1 --per-event 1000 --seed-offset 41` 和 `scripts/data/build_confirm_direct.py`。影像全部通过时间核验（220/220，289/289）。

## 3. 冻结的运行设置

| 运行 | 模型 | 证据 | 提示词版本 | max_tokens |
|---|---|---|---|---|
| `confirm_pairs_sonnet` | claude-sonnet-5-5 | image, meta | pairs-v1-2026-10-19 | 3000 |
| `confirm_pairs_qwen38` | Qwen3.8-27B | image, meta | pairs-v1-2026-10-19 | 12000 |
| `confirm_direct_sonnet` | claude-sonnet-5-5 | image, image_mismatch（neutral） | v1-2026-10-19 | 3000 |
| `confirm_direct_qwen38` | Qwen3.8-27B | image, image_mismatch（neutral） | v1-2026-10-19 | 12000 |
| `confirm_direct_haiku` | claude-haiku-5-5 | inspector（neutral） | v1-2026-10-19 | 3000 |

- 采样：Claude 使用 API 默认采样（该模型不接受 temperature 参数），Qwen 使用 temperature=0。每个条件只跑一遍。
- 重试：未解析成功的样本最多重试 2 次，每次 max_tokens 翻倍。之后仍失败的记为缺失并报告缺失率。若任一条件缺失率超过 5%，另报"缺失记为错"的敏感性结果。
- 分析代码：`scripts/eval/confirmatory_analysis.py`，以及 `scripts/eval/confirm_references.py`（参照模型只在 train 事件上拟合，不用 val）。

## 4. 假设与判定

主要检验族为 H2、H3、H4a，使用单侧 α=0.05 并做 Holm 校正。主区间用配对或单栋层面的 bootstrap（2000 次）。val 只有 9 场火且 Glass 占 61%，事件聚类 bootstrap 不稳，因此作为敏感性分析报告，同时报告去掉 Glass 后的结果。

| 编号 | 假设 | 指标 | 检验力（按测试集效应估计） | 判定 |
|---|---|---|---|---|
| **H2**（主要） | Sonnet 看灾前图做邻居对比，优于随机 | 配对准确率 > 0.5（p=0.5 计半分） | 真实值为 0.62 时约 0.99 | 拒绝原假设，则"VLM 在暴露控制下有建筑层面信号"升为一级 |
| **H3**（主要） | Sonnet 直接判断时利用了图像 | AUROC(本栋图) − AUROC(错配图) > 0 | 效应为 +0.068 时约 0.85 | 拒绝原假设，则"图像增量真实"升为一级（需注明效应小） |
| **H4a**（主要） | Haiku 读巡查员属性时方向反转 | 全集 AUROC < 0.5 | – | 与 H4b、H4c 一致，则缺失机制升为一级 |
| H4b（描述） | 只看属性齐全的子集时反转消失 | 子集 AUROC | 子集可能很小 | 只做描述 |
| H4c（描述） | Unknown 个数与结局正相关，与 Haiku 的预测负相关 | 两个相关系数 | – | 只做描述 |
| **H1**（估计，不做检验） | Sonnet 配对中图像比仅元数据更好 | Δ 准确率及 95% CI | **只有约 0.37**（Δ=0.039，n=289，最小可检出效应 0.073） | 只报告估计值和区间。区间含 0 **不能**解读为"图像无增益"；另报与测试集 403 对合并的估计（n≈690） |
| H5（探索） | Qwen 直接判断时的图像增量 | 同 H3 | – | 报告，并与 Sonnet 并列 |
| H6（次要） | Sonnet 图像配对准确率与只用 train 拟合的表格元数据模型比较 | 两者之差（描述） | – | 描述 |
| H7（次要） | 嵌入擅长判断位置而非区分建筑 | DINOv2 头的 AUROC 明显高于其配对准确率，后者接近 0.5 | – | 描述 |

## 5. 不会做的事

- 看到结果后不修改提示词、阈值、解析规则或样本。
- 不因结果不理想而追加模型或条件。若追加，一律标为探索性分析。
- 不把 H1 的不显著写成"图像无用"。
