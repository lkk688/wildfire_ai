# SJSU 野火跨学科研究中心（WIRC）调研及其与本项目的关系

调研日期：2026-10-22。来源包括 wildfirecenter.org 全站（按 sitemap 抓取，含未出现在菜单里的页面）、NSF IUCRC 页面、NSF Award API、GitHub、Zenodo 和 OpenAlex。本文标注了核实程度：**[已核实]** 表示我们读过原文或原图；其余条目来自整理后的检索结果，引用前应再核对原文。

---

## 1. 机构概况

- **全称**：Wildfire Interdisciplinary Research Center（WIRC），设在 San José State University，是 **NSF 产学合作研究中心（IUCRC）**，也是 CSU 系统的正式研究中心。
- **主任**：Craig B. Clements（SJSU 气象与气候科学系主任、Fire Weather Research Lab 创始人）。WPI 站点主任为 Albert Simeoni。联合主任 / 研究主任为 Julia Gaudinski。
- **NSF 资助**：
  - 2113931，SJSU 主站，2021-08 至 2028-07，已拨款约 133 万美元；
  - 2333294，WPI 分站，2023-09 至 2028-08，已拨款约 60 万美元。
  - 这是 NSF 中仅有的两笔 WIRC 资助。俄勒冈大学、科罗拉多大学丹佛分校和各国际合作方都只通过个人关联，**不是 IUCRC 站点**。
- **会员模式**：行业顾问委员会（IAB）会员费为 $50,000/年，只有一档。成员包括 CAL FIRE、PG&E、SDG&E、SCE、NOAA、NASA、CSAA、Mercury Insurance、RenaissanceRe、FSRI、Technosylva、Lockheed Martin、Watch Duty、Earth Fire Alliance 等。会员资金按年度资助项目，并享有中心知识产权的免版税、非独占许可。
- **主要人员**：
  - SJSU：Adam Kochanski（WRF-SFIRE 模型、Fire Modeling Group）、Kate Wilkin（火生态、住宅周边易燃区 HIZ 与可防御空间）、Minghui Diao（烟雾遥感）、A.J. Faas（人类学）、Ben Reed（计算机）、Aurélien Costes 与 Angel Farguell（建模研究员，FireBench 主力）。
  - WPI：Simeoni、James Urban（WUI 点燃、传感、计算机视觉）、Carlo Pinciroli（机器人）等。
  - 关联学者：Amanda Stasiewicz（俄勒冈大学，社会科学 / 疏散）、Ali Tohidi（马里兰大学，飞火）、Bo Yang（UCSC，无人机 / GIS）、Jan Mandel、Mario Valero。
- **算力与设施**：Ember CPU 集群（1,120 核，含 2 个 A100 节点）；6 节点 GH200 集群，用于"AI 蔓延速率模型"；Ka 波段移动多普勒雷达、多普勒激光雷达；Geofly 无人机实验室（可见光、多光谱、热红外、LiDAR）。

## 2. 研究路线图（三大方向）

| 方向 | 子方向 |
|---|---|
| Modeling（建模） | 火灾天气、火行为、野火烟雾、WUI、气候变化与野火风险 |
| Resilience（韧性） | 野火管理与政策、WUI 适应、材料耐火性 |
| Operations（运营） | 野火作业、消防员安全 |

## 3. 与本项目直接相关的工作

### 3.1 FireBench：模型评测框架，其中包含基于 DINS 的建筑损毁基准 [已核实]

- **框架**：与具体模型无关的火灾模型评测框架。观测和模型输出统一为 HDF5 加 KML 格式；按 KPI 打分，每项归一化到 0–100，再汇总成总分；结果可数字签名，支持第三方认证；IAB 会员可以提交自己的模型参与排名。
  - 代码：https://github.com/wirc-sjsu/firebench（Apache-2.0，最新版本 v0.10，2026-08）
  - 文档：https://firebench.readthedocs.io
- **2021 Caldor 基准（FB001）**：数据 DOI 10.5281/zenodo.23383592，v2026.3，2026-10-15 发布。观测包括：
  - CAL FIRE **DINS** 建筑损毁（2025-11-05 版，4,444 栋）；
  - MTBS 烧毁严重度、RAVG 林冠损失、LANDFIRE 林冠参数；
  - 21 条 NIROPS 红外火线、气象站数据。
- **二分类口径与我们完全一致**：burnt = Major + Destroyed；unburnt = No Damage + Affected + Minor；Inaccessible 不计。KPI BD01–BD06 分别是准确率、精确率、召回率、特异度、阴性预测值和 F1。
- **示例结果（从 NSF 页面图 2 读取）**：WRF-SFIRE 预报在建筑损毁上的准确率 **0.248**、召回率 **0.989**、F1 **0.375**。按 Caldor 约 23% 的损毁率反推，特异度约 **0.03**。
  - 也就是说，**物理蔓延模型把火场内几乎所有房屋都判为烧毁**，与我们在 LLM 身上测到的"Everything Burns"是同一种失败模式，只是成因是格点分辨率：每栋房子直接继承所在格点的燃烧状态。
- **与本项目的关系**：
  - 他们评测的是**火蔓延模型**能否复现建筑损毁，而不是建筑本身的脆弱性。
  - 我们的结论正好可以补上这一层：在暴露条件已经确定的情况下，建筑层面的信号上限只有约 0.62–0.65。
  - 两者合起来可以说明：目前的物理模型解决"火到没到"，建筑层面的差异仍然很难预测。
  - 2026 年 IAB 项目"Probabilistic Framework for Fire Risk Assessment using Fire Spread Models in FireBench"表明他们正往风险评估方向扩展，是潜在的对接点。

### 3.2 住宅周边易燃区（HIZ）与可防御空间合规性研究

- 4 年内共 4 个 IAB 项目。PI 为 Wilkin、Stasiewicz、Yang。导师来自 CSAA 和 PG&E。
- 样本：5 个社区共 176 户，分别为 Felton、Redwood Estates、Saratoga、Paradise、Tahoe Donner。
- 方法：先入户检查，再测试无人机和机载 LiDAR 能否扩大检查规模，并辅以访谈。
- 发表：
  - Wilkin, Benterou & Stasiewicz (2025), *IJDRR* 124:105435。结论：高风险 WUI 住宅的强制和自愿措施完成率都很低，远低于官方统计和住户自报的水平。
  - Benterou 硕士论文 (2026)：屋顶和排水沟的积叶主要由树冠覆盖驱动，光滑金属屋顶最不易积叶，排水沟护网能减少积叶。
- **关键原话（NSF 页面）[已核实]**："Airplane and satellite data … their spatial resolution is still not high enough to accurately delineate the 0-5 foot zone or detect small features"。他们认为无人机检查更快、更标准化，还能看到屋顶隐患；机载 LiDAR "promising"。
  - **这与我们的结果相互印证**：在 0.3–0.6 m 航拍上，细粒度属性不可判定（Qwen 的旋转一致性只有 58–66%，Sonnet 判屋顶材料不如全猜多数类）。
- 尚未发表：
  - Brillon, Wilkin & Yang，*Inspecting residential wildfire hazards using UAS and object-based machine learning*（在审）；
  - Stevens 等，LiDAR 可防御空间（在写）；
  - Lackey 等，*Discrepancies between Zillow and property records*（在写）。

### 3.3 与 WUI 建筑相关的其他项目（多数只有标题，未找到产出）

- 2024：Drone Inspection for Home Ignition Zone（Bo Yang）；Post-Wildfire Fuels Management using Multi-source Remote Sensing。
- 2025：**Home Ignition Zones: Building footprint, build year, and related codes**。建造年份正是我们灾前信号最强的特征。
- 2025–2026：WUI fire exposure to structures from near-home combustibles: Firebrands, Flame Contact, & Radiation（WPI 牵头）。
- 2026：Wildfire Data Exchange Platform；Expertfluencers（减灾市场网络）；输电设备概率风险。

### 3.4 建模、遥感和 AI（与我们关系较远）

- WRF-SFIRE 同化：Kochanski 2023；Clough 2025 JGR。
- 生成模型重建火线：Shaddy 2024 AIES，用 cWGAN 从卫星热点推断到达时间。
- 燃料含水率：活燃料 30 m 机器学习气候态；Farguell 2024 年死燃料再分析。
- 机器学习蔓延速率模型（2023 起）。
- AI 气象框架（2026，GitHub 上有 GraphCast / FourCastNet 分叉）。
- Brown 等 2023 年 *Nature*：气候变暖增加极端日增长风险。
- Ren 等 2026 年：无人机热红外和燃料结构。
- **全站没有任何视觉-语言模型（VLM）相关工作。**

## 4. 与本项目关系最密切的外部论文（WIRC 关联学者合著，但不属于 WIRC 产出）[已核实]

**Zamanialaei, …, Tohidi, …, Gollner (2025). *Fire risk to structures in California's Wildland-Urban Interface*. Nature Communications 16:8041.** https://doi.org/10.1038/s41467-025-63386-2

- 数据：DINS 中 5 场火（Tubbs、Thomas、Camp、Kincade、Glass），约 4.7 万栋。
- 特征：
  - 建筑间距（来自 Microsoft US Building Footprints）；
  - 0–5、5–30、30–100 ft 三圈植被（航拍和 LiDAR；只有 3,504 条非空）；
  - DINS 建筑属性；
  - 用 ELMFIRE 复盘真实火灾得到的火焰长度和飞火量。
- 模型：XGBoost，准确率 82%，AUC 0.83；最重要的特征是建筑间距。
- **评测协议**：原文写的是"allocating 80% of the data for training … remaining 20% for the testing set"，即五场火混在一起随机划分，不按火灾划分，也不按时间。
- **属性来源**：使用灾后 DINS 属性，缺失值用多种方法插补。作者承认"inherent uncertainty on collected features that are difficult to determine in a post-damage state"。
- 暴露特征：火焰长度和飞火量来自对**已发生火灾**的复盘，不是灾前可知的信息。

**我们的数据对这篇论文的含义**（只是推断，尚未直接复现）：

| 我们的发现 | 对这篇论文的含义 |
|---|---|
| 随机划分会抬高分数：灾前元数据 0.551 → 0.712 | 82% / AUC 0.83 很可能偏乐观 |
| 只用"哪些属性缺失"就能达到 0.883 | 插补没有消除缺失本身携带的信息 |
| 密度规则在 2019–23 年乡村火和 2025 年城市火之间方向相反（AUROC 0.391） | "建筑间距是首要因素"未必能跨事件成立 |

这些是方法学上的风险，不是对其结论的否定。如果要严格比较，应在我们的事件和时间划分下复现他们的特征集，可以用 Microsoft 建筑轮廓计算间距。

## 5. 对本项目的启发与可做的事

1. **定位**：WIRC 在火天气、蔓延模型、评测基准和住宅合规社会科学方面很强，**但在"影像 / AI 判定建筑脆弱性"和"评测协议本身"上是空白**。我们的诊断论文（灾前不可预测、灾后可判定、评测协议导致虚高）可以和 FireBench、HIZ 研究形成互补，而不是竞争。
2. **可以直接用的数据和工具**：
   - FireBench 的 Caldor 基准，与我们的 DINS 二分类口径一致，可作为外部对照；它包含的 MTBS 严重度也可作为暴露代理。
   - FireBench 的 KPI 和打分卡格式，可以把我们的模型（如灾后 SFT）按他们的格式提交、获得排名。
   - WIRC 的燃料含水率再分析和 WRFx 天气数据，可补上我们缺的"真实事件天气"条件。
3. **可以合作的方向**：
   - Kate Wilkin / Bo Yang 有 176 户**入户实测**的 HIZ 合规数据，外加无人机和 LiDAR。这正是我们缺的独立真值，可以核验 VLM 从航拍提取的属性。他们已经发现航拍和卫星分辨率不够，我们的结果可以量化"差多少"。
   - Costes / Kochanski 维护的 FireBench 可以纳入建筑脆弱性模型，与蔓延模型的输出耦合（概率风险框架），即"蔓延模型给暴露，我们给建筑层面的条件风险"。
   - IAB 中的保险公司（CSAA、Mercury、RenaissanceRe）是保险合规和弃答可靠性研究的天然用户。
4. **需要引用和对比的文献**：Zamanialaei 2025 *Nat Commun*；Wilkin 2025 *IJDRR*；FireBench（Zenodo / GitHub）；Gaudet、Simeoni 等 2020 年对 WUI 灾后调查研究的综述。
