# 项目交接文档 — Lagrangian Cuts 复现与拓展

> 面向:接手本项目、在**本地机器**继续开发的人（含 Claude 桌面客户端）。
> 目标:读完这份就能 clone 下来跑起来，并清楚接下来该做什么、为什么。

---

## 0. 一分钟速览

- **做了什么**:从零复现了论文 Chen & Luedtke (2022), *On Generating Lagrangian
  Cuts for Two-Stage Stochastic Integer Programs* (arXiv:2106.04023v2)，并对其
  “受限分离(restricted separation)”做了方法层面的拓展探索。
- **复现**:完整、已验证、已推送。论文六种割生成法都实现了，理论关系和核心
  收敛图都对上。
- **拓展探索的核心结论(重要)**:
  1. **论文定理 3 证明所有第一阶段割方法给够时间都收敛到同一个界 `z_D`。**
     所以任何“换 basis”的技巧**只能提速，不能提界**。
  2. 小预算下 basis 质量确实影响收敛界，但赢家是论文自己的 **rstrmip**
     (按“当前违反量”选方向)——我提的 **adaptiveA**(SVD 主成分选方向)**输了**。
     教训:**选方向要“有的放矢”,不能按“统计主流”。**
  3. 想突破 `z_D` 天花板、逼近真实最优 `z_IP`,**只有“析取割(disjunctive
     cuts)”这条路**(见下方方向 2)。
- **现实限制**:云端用的是开源 HiGHS，很多“跑到完全收敛”的对照实验会撞时间墙。
  **本地 Gurobi 会快很多**,这也是转到本地的主要动机。

---

## 1. 仓库与分支

- GitHub 仓库:`jinshuangya/Paper_Implement`
- 开发分支:`claude/paper-reproduction-method-extension-orc8dx`

本地获取:

```bash
git clone https://github.com/jinshuangya/Paper_Implement.git
cd Paper_Implement
git checkout claude/paper-reproduction-method-extension-orc8dx
```

---

## 2. 环境与运行

```bash
pip install -r requirements.txt          # numpy scipy highspy matplotlib
PYTHONPATH=src python -m pytest tests -q  # 验证(约 30s)
```

**求解器**:目前所有 LP/MIP 都走开源 **HiGHS**(`highspy`)，无 license、跨平台。
瓶颈是反复求解单场景整数子问题(下称 **oracle**)。

> **强烈建议在本地接一个 Gurobi 后端。** oracle 的 MIP 用 Gurobi 会快数倍，
> 云端很多 delta=0 完全收敛实验因此撞时间墙、拿不到干净结论。做法:把
> `src/lagcuts/oracle.py`、`separation.py`、`rstrmip.py`、`master.py`、
> `reference.py` 里对 `highspy.Highs()` 的构造抽到一个统一的求解器工厂,
> 按环境变量或自动检测 `gurobipy` 切换后端。接口很薄(建变量、加约束、
> 取解与对偶),改动集中、风险低。**这是接手后建议做的第一件基础设施。**

---

## 3. 术语速查(看实验输出前先扫一眼)

| 缩写 | 含义(大白话) |
|---|---|
| **oracle** | 那个“贵”的单场景整数子问题 `Q*_s(π,π0)`(论文式 12)。反复解它是主要耗时。实验里 `oracle` 数 = 解了多少次,越少越好。 |
| **LP 界** | 最弱的起步下界(松弛版给的)。 |
| **z_D** | 这套 Lagrangian 割**理论上的最强界**(论文定理 3 的天花板)。 |
| **z_IP** | 问题的**真实最优解**。一般 `z_D ≤ z_IP`。 |
| **benders** | 只加便宜的 Benders 割 → 只到 LP 界。 |
| **exact** | 在完整方向空间搜割,最强但最慢。 |
| **rstr1 / rstr2** | 论文加速法:在“最近 K 条 Benders 割”张成的小空间搜(两种归一化,式 19/20)。 |
| **rstrmip** | 论文最好的方法:用一个 MIP(式 28)**按“此刻最能切中当前点”从所有 Benders 割里挑 K 个方向**。 |
| **adaptiveA** | *本项目的拓展尝试*:候选方向 = Benders 方向 ∪ oracle 吐的整数顶点 `x*`,用 **SVD** 抽前 K 个主方向。**结论:输给 rstrmip,机制不对。** |
| **rstrmipV** | *拓展的“纠偏版”*:保留 rstrmip 的 MIP 选择器,但候选池加入整数顶点 `x*`。**尚无定论(需 Gurobi 跑到收敛)。** |
| **K** | 搜索小空间的维度(预算)。越小越快、越可能不够用。 |
| **delta** | 每条割的求解精度容差。`delta=0` = 死磕到完全收敛(慢)。 |
| **stop_reason** | 算法为什么停:`no_cut`=真收敛(找不到违反割了);`time_limit`=撞时限(**未收敛,数值不可当收敛值用**)。 |

---

## 4. 代码结构

```
src/lagcuts/
  sslp.py         SSLP 实例生成(论文附录配方);capacity_scale 与 decoupled 两个拓展旋钮
  oracle.py       Q_s(x)、Benders LP 对偶割、齐次值函数 Q*_s(π,π0);第二阶段约束集中在 _add_second_stage
  separation.py   Algorithm 1 受限分离 + 上界模型 Qbar*_s + 受限域 Π_s(exact/rstr1/rstr2)
  rstrmip.py      RstrMIP 选基 MIP(式 28)
  adaptive.py     拓展 A:整数顶点 + SVD 的自适应子空间(结论:机制不对)
  master.py       Benders 主问题(根节点 LP 松弛),动态加割
  algo2.py        Algorithm 2 主循环;方法开关 method= 见下;记录 LB-时间轨迹与 stop_reason
  reference.py    extensive-form z_IP(验证用)
experiments/
  run_rootnode.py       根节点“界 vs 时间”对照图(复现主图)
  run_basis_rank.py     诊断:Benders 方向张成秩 vs 整数顶点秩,随 LP 变松的坍缩
  run_extensionA_demo.py 拓展 A 的收敛界对照(delta=0),标注 stop_reason
tests/test_validation.py  理论序关系验证
docs/figures/             已复现的图(见下)
```

`algo2.Config(method=...)` 支持:`benders` / `exact` / `rstr1` / `rstr2` /
`rstrmip` / `adaptiveA` / `rstrmipV`。

---

## 5. 已复现的结果(附图)

三张图都在 `docs/figures/`:

- **`bound_vs_time_sslp1_10_25_30.png`** — 复现论文主结论:rstr2/rstrmip 最快闭合
  Lagrangian 对偶间隙,rstr1 因 basis 受限略低,exact 明显慢(时限内没收敛)。
- **`basis_rank_collapse.png`** — 拓展 A 的结构性证据:LP 变松时 Benders 方向张成
  秩从 9 掉到 ~5,而整数顶点张成始终满秩 10。
- **`bound_vs_time_sslp1_8_15_15.png`** — 较小实例的同类对照。

复现关键数字(m=10,n=25,S=30):LP 界 −292.5 → rstr2/rstrmip 100% 闭合到
−149.8;exact 90s 内只到 −153.9(未收敛);oracle 调用 rstr2/rstrmip(773)< exact(2019)。

---

## 6. 拓展探索的完整轨迹与结论(这段最值钱)

**动机(方向 1)**:论文的搜索子空间只由 Benders(LP)方向构成;LP 一差,方向就差。
设想掺入整数子问题给的方向会更好。

**做过的实验与所得:**

1. **adaptiveA(整数顶点 + SVD)** 能突破 rstr1 的“滑动窗”受限平台、追平 z_D,
   但在正常实例上只是追平、oracle 还略多。
2. **秩坍缩机制是真的**(`basis_rank_collapse.png`):造了 `decoupled`(松链接+紧容量)
   实例族,LP 松时 Benders 方向秩坍缩,整数顶点满秩。
3. **撞到 z_D 天花板(定理 3)**:满预算 + delta=0 下,rstr1/rstr2/rstrmip/adaptiveA
   **全部收敛到完全相同的 z_D**。→ **basis 技巧只能提速,不能提界。**
   (我曾一度被 `time_limit` 未收敛的假象误导以为出现了“界差距”,后被 delta=0
   收敛实验推翻。**看结论务必确认 `stop_reason=no_cut`。**)
4. **固定小预算 K=3、delta=0 全收敛**的公平对照(能区分 basis 质量):
   | 方法(K=3) | 收敛界 | 到 z_D 距离 |
   |---|--:|--:|
   | rstr1 | −29.79 | 差 24 |
   | rstr2 | −15.97 | 差 10 |
   | **rstrmip** | **−5.75 (=z_D)** | **0** |
   | adaptiveA | −8.20 | 差 2.4 |

   **结论:小预算下 basis 质量确实决定收敛界,但赢家是 rstrmip,不是 adaptiveA。**
   原因:rstrmip 按“当前违反量”选方向(有的放矢),adaptiveA 的 SVD 按“统计主成分”
   选(瞎选)。**violation-targeted 完胜 variance-targeted。**

**净结论**:adaptiveA 作为当前设计**不成立**。但它教会我们两条硬道理:
(a) 界有 z_D 天花板,换 basis 只能提速;(b) 选方向必须“有的放矢”。

**尚未定论的一步**:`rstrmipV`(= rstrmip 的 MIP 选择器 + 候选池加整数顶点)——
理论上是“方向 1 的正确形态”,但云端 delta=0 一直撞时限,拿不到干净收敛值。
**这是留给本地 Gurobi 的第一个实验(见下)。**

---

## 7. 接下来的建议(按优先级)

> **2026-09 更新**:本节的方向已在 [`TOPIC_TRIAGE.md`](TOPIC_TRIAGE.md) 中
> 与另一份《顶刊选题与研究计划》的全部 ~30 个 Topic 一起重新筛选过,
> 约束是「无 GPU / 无 LLM API 经费 / 由 AI agent 执行」。结论:下面的 **A3**
> (学一个网络摊销 oracle)被排为第一顺位,理由是**论文定理 3 白送了一个
> 「学习组件不可能损害界」的安全性定理**。可执行的实验规格见该文档 §3.1。


研究目标其实分两类,因为 z_D 是天花板:

### A 类:更快到达 z_D(加速)——原方向 1、4,需纠偏
- **A1. 先把求解器 Gurobi 后端做了**(见第 2 节),否则收敛实验跑不动。
- **A2. 跑通 `rstrmipV` 的判定实验**:在**强坍缩**的 decoupled 实例上、**K 给到 ≥ m**
  (让瓶颈落在“候选池 span”而非 K 预算)、**delta=0 真收敛**,比较 rstrmip
  (只有 Benders)与 rstrmipV(Benders ∪ 整数顶点)。
  - 若 rstrmipV 收敛界 > rstrmip → “更丰富候选 + 违反量精选”成立,方向 1 有救。
  - 若持平 → 整数顶点对 MIP 选择器无增量,需升级到**整数割的法向**
    (Gomory / MIR / lift-and-project,投影回 x 空间)作为候选——这才是你原始
    Gomory/MIR 直觉的正确落点。
- **A3. 方向 4 纠偏版**:别去替换那个便宜的选基 MIP;要**学一个网络(GNN/监督式)
  模仿 rstrmip 的“按违反量选方向”**,把真正贵的 **oracle 摊销**掉。训练目标明确
  (模仿 rstrmip 的选择),随机规划天然有实例分布可训。

### B 类:突破 z_D、逼近 z_IP(提界)——原方向 2,理论天花板最高
- **B1. 析取割(disjunctive cuts)**是**唯一**能越过 z_D 的路(它不是式 13 那族割,
  不受定理 3 约束)。最小可行实验:用 rstrmip 高效闭合到 z_D 后,对残余的分数解
  加一层析取割/lift-and-project,看能否把界推过 z_D、朝 z_IP 走。
- 注意选题避免撞车(Qi & Sen [12]、van der Laan & Romeijnders [14] 已有混合框架)。
  更新颖的理论靶子:**自适应受限分离方案(Π_s 随迭代变化)的收敛性刻画**——论文
  定理 5 只覆盖静态 Π_s。

### 暂缓
- **方向 3(多阶段 SDDIP)**:工程量大、且论文已点名为 future work,novelty 天花板低。
  留作 A/B 成熟后的规模化载体。

---

## 8. 给本地接手者的第一批具体 TODO

1. [ ] 加 Gurobi 可切换后端(第 2 节),`pytest` 全绿。
2. [ ] 跑 A2(`rstrmipV` 判定):强坍缩 decoupled 实例 + 大 K + delta=0 真收敛,
       确认是否超过 rstrmip;截图与数字记录到 `docs/`。
3. [ ] 若 A2 持平,实现 Gomory/MIR 割法向作为候选池(A2 的升级)。
4. [ ] 起草 B1 析取割最小实验(唯一能提“界”的方向)。

> 复现部分已定稿可信;上面 6、7 两节是本次探索的真正产出——**尤其是“z_D 天花板 +
> 违反量选择完胜 SVD”这两条,直接决定了拓展该往哪走。**
