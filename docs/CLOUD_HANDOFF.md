# 云端交接：跑完已登记的跨域账本

写给克隆了 `https://origin.cursor.com/leo-vino/QuantLab.git` 分支 `main` 的云端 Agent。先读本文，再读 `docs/prereg/P4_DOMAIN_ALLOC.md`。预注册登记后不可改。

更新时间：2026-10-15 16:46 CST。`promotion_approved=false`。

## 要做的事

把已经生成的 12 条持仓路径，各回放成本 1 倍和成本 2 倍，共 24 本。写完后出报告，把结果推回你的云端分支。

这是已登记试验 `P4_domain_alloc` / `XDOM`。`XDOM_RAND` 是诊断对照。`docs/trial_ledger.csv` 里这两行已经是 `pending`，不要再登记一次，也不要改预注册的 sha256。

## 仓库里已经有的

- 目标持仓：`replay_data/p4_domain_alloc/plans.json`
- 各域只数：`replay_data/p4_domain_alloc/holdings_mix.json`
- 行情：`replay_data/p4_domain_alloc/dataset/manifest.json` 及其中 1217 只面板、交易日历、证券列表、公司行为、基准
- 已公布的 `POOL50` 对照：`replay_data/p4_domain_alloc/reference/pool50/`（两份 summary，以及 8 条成本 1 倍净值）
- 低换手参照：`replay_data/p4_domain_alloc/reference/hyst/report.json`（成本 1 倍 14.38%，只作报告，不是通过条件）

`scripts/p4_domain_alloc.py` 的账本阶段在这份 `plans.json` 存在时直接读它。不要跑 `plans`。那一步会去读本地的 `output/p4_domain_20261014/scores.parquet` 和 `samples.parquet`，克隆里没有这两份，也不要去找或重做。

## 不要做的事

- 不重训，不重造样本，不改分档数、滞后比例、切点 0.40/0.70、名额 17/17/16。
- 不接 `quantlab/strategy/allocation.py`，不做市值中性。
- 不下载 `data/` 下约 16GB 的原始行情。这次账本不读它。完整行情和历史实验产物在 `https://origin.cursor.com/leo-vino/QuantLab-data.git`，这次回放不要克隆那个仓。
- 不补写旧试验 `P4_domain` 没完成的四本成本 2 倍：`DOM_RAND_p10_s2`、`DOM_GBDT_p15`、`DOM_RAND_p15_s1`、`DOM_RAND_p15_s2`。它们不在本包里。
- 不把 `promotion_approved` 改成 true。通过也只表示 `XDOM` 可以成为下一层配比的候选，要另写预注册才能换纸面账。
- 账户检查失败就停在 `technical_failure`。不要改规则重跑。门槛不过或退回低成交，就记 `no_improvement`，股票账维持 `POOL50`，不要再调进出比例或分档。

目标持仓已经偏向低成交（模型四相位大约 48 只在低成交）。这是预注册允许的结果。先把 24 本跑完，用报告里的门槛判定，不要因为持仓结构提前停掉。

## 环境

在仓库根目录：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 回放前放好对照账

报告脚本从 `output/` 读已公布的 `POOL50`，那个目录不在 git 里。把对照文件放到它期待的路径，不要重算 `POOL50`：

```bash
mkdir -p output/p4_alpha_20261012/ledger output/p4_lowturn_20261013
cp replay_data/p4_domain_alloc/reference/pool50/* output/p4_alpha_20261012/ledger/
cp replay_data/p4_domain_alloc/reference/hyst/report.json output/p4_lowturn_20261013/report.json
```

## 24 条账本

每条路径跑两次，`--cost 1` 和 `--cost 2`。可以并行，每条命令只跑一个路径。路径名是：

- `XDOM_p0` `XDOM_p5` `XDOM_p10` `XDOM_p15`
- `XDOM_RAND_p0_s1` `XDOM_RAND_p0_s2`
- `XDOM_RAND_p5_s1` `XDOM_RAND_p5_s2`
- `XDOM_RAND_p10_s1` `XDOM_RAND_p10_s2`
- `XDOM_RAND_p15_s1` `XDOM_RAND_p15_s2`

```bash
.venv/bin/python scripts/p4_domain_alloc.py ledger --cost 1 --only XDOM_p0
.venv/bin/python scripts/p4_domain_alloc.py ledger --cost 2 --only XDOM_p0
```

其余路径把 `--only` 换成上面的名字。同时跑的进程不要超过 4 个。最早的调仓日 2018-01-02 可以是空仓，标签还没走完，这是预注册写过的。

净值写在 `output/p4_domain_alloc_20261015/ledger/`。这个目录被 git 忽略。24 本都完成后，把净值和分段结果复制出来再提交：

```bash
mkdir -p replay_data/p4_domain_alloc/ledger
cp output/p4_domain_alloc_20261015/ledger/nav_*.csv output/p4_domain_alloc_20261015/ledger/part_*.json replay_data/p4_domain_alloc/ledger/
```

## 报告

```bash
.venv/bin/python scripts/p4_domain_alloc.py report
```

脚本会把 `replay_data/p4_domain_alloc/report.json`、`summary_cost1.csv`、`summary_cost2.csv` 写好，并在 `docs/trial_ledger.csv` 追加结果行。缺任何一条路径时它会退出 `incomplete_ledger`，这时补跑缺失的路径，不要改规则。

四条门槛都在预注册里，必须同时满足才算通过：

1. 成本 1 倍年化中位数比 `POOL50` 高至少 2 个百分点。已公布的 `POOL50` 成本 1 倍中位数是 12.11%。
2. 相位 0 对 `POOL50_p0_s1` 的日超额，Newey-West t（滞后 20）不低于 2。
3. 成本 2 倍年化中位数仍高于 `POOL50` 成本 2 倍中位数。
4. 2022-01-04 至 2026-09-22 的年化中位数高于 `POOL50` 同段。

四个相位的平均低成交只数都不低于 40，并且成本 1 倍中位数不超过 `POOL50` 时，结果是 `collapsed_to_micro`，结局仍是 `no_improvement`。

## 写回仓库

在 `docs/RESEARCH_CONTINUITY.md` 顶部新增一节，不要改旧段落。用简体中文写明：四条门槛各自是否通过、成本 1 倍和 2 倍年化、相对 `POOL50` 的差距、2022 年以后的年化、Newey-West t、低成交平均只数、结局、`promotion_approved` 仍为 false。

提交并推送这些文件：

- `replay_data/p4_domain_alloc/report.json`
- `replay_data/p4_domain_alloc/summary_cost1.csv`
- `replay_data/p4_domain_alloc/summary_cost2.csv`
- `replay_data/p4_domain_alloc/ledger/` 下的净值和分段结果
- `docs/trial_ledger.csv`
- `docs/RESEARCH_CONTINUITY.md`

不要提交 `output/`、`data/`、虚拟环境，也不要提交密钥。
