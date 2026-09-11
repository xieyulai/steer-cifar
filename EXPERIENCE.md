# EXPERIENCE — 实验经验记录

## 精华摘要（滚动，compress 维护）

- **当前 SOTA（来自 _runs/results.tsv）**: `20260910_183749_682528_s1of2_r10_tierC_ls_sweep_s1_ls01` · val_accuracy=0.977
- **Plateau**: focus=cifar10_autonomous plateau=3/3 std=0.0331727
- **Tier 快照**: A-r:attested×4（R13/R14/R19 LR sweep 0.05/0.03/0.08 + R12 SAM_RHO 0.03/0.05, R19 s1 LR=0.08 0.9752 验证 R10 真 optimum） B-r:浅尝→attested×3（R11 PreActResNet18 0.9669 + R20 s1 WRN-40-4 100ep smoke 0.9698 + R21 s0 WRN-40-4 200ep full 0.9715, 三档全证实 capacity-bound 不破 0.977） C-r:浅尝（LS sweep attest，2 值） D-r:浅尝→attested（R14 s1 reference_anchor mixup-only 0.9720, baseline_tag=reference 落… E-r:未试
- **本档要点**: A-r:attested×4（R13/R14/R19 LR swe…；B-r:浅尝→attested×3（R11 PreActResNe…；C-r:浅尝（LS sweep attest，2 值）；D-r:浅尝→attested（R14 s1 reference_…
- **preflight 行占比**: 0.0%
- **本轮归档实验段**: 17 条（全文见 `references/experience/archive_*.md`）
- **近端台账（摘要）**:
  - `20260911_120241_1316765_s1of2_run`: val_accuracy=0.975200
  - `20260911_141040_3003715_s0of2_run`: val_accuracy=0.975800
  - `20260911_141040_3003716_s1of2_run`: val_accuracy=0.969800
  - `20260911_162749_672711_s0of2_r21_tierB_routine_s0_wrn40_4_200ep`: val_accuracy=0.971500
  - `20260911_163017_706332_s1of2_r21_tierA_novel_s1_lookahead`: val_accuracy=0.937600
- **REFLECT pending**: `R20260911_161804` — （人审 E）Promote to Tier E: run PC-DARTS-style micro-cell search on the wrn28_10 s… (tier=A)

## 信息索引（按需 Read，默认不展开）

| 类型 | 路径 |
|------|------|
| 反思 pending | `references/REFLECT_INDEX.md` |
| 反思详文 | `references/auto/<reflect_id>_auto_reflected.md` |
| 外部证据 | `saved/evidence_bundle.json`、`saved/external_evidence/pdfs/`、`saved/reflect_evidence.json` |
| 压缩归档 | `references/experience/archive_*.md` |
| 台账 | `_runs/results.tsv` |
| keeper | `saved/keepers.json` + `_runs/exp/<keeper>/config.json` |

## 场景与 KEEP 约定（滚动更新）
> 表行 **EVAL_SCENARIO** 须与 F1「场景清单」中的场景 ID 一致；KEEP substr 列与清单 KEEP 列一致。

| EVAL_SCENARIO | 主指标键 | 当前最佳 | 最佳 experiment | KEEP 对照 substr | 下一轮计划 |
|---------------|----------|----------|-----------------|------------------|------------|
| cifar10_autonomous | val_accuracy | 0.9770 | 20260910_183749_682528_s1of2_r10_tierC_ls_sweep_s1_ls01 | plain | wrn28_10 + 复合 cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1；R12 双槽 SAM_RHO ∈ {0.03, 0.05} on R10 KEEPER 双双 DISCARD (time_bu… |
| cifar10_source_reference | val_accuracy | 0.9585 | 20260826_145142_3232878_s0of1_reference_dla_rerun | reference | 已锁定 public-anchor；本场景不在 focus |

**策略（人话一行）：** focus=cifar10_autonomous KEEPER wrn28_10 + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 → 0.9770；R12 双槽 SAM_RHO ∈ {0.03, 0.05} on R10 KEEPER 双双 DISCARD（time_budget 截断 epoch 107/200, s0=0.9379, s1=0.9485）— SAM 2× compute 撞 7200s 预算天花板，cosine LR 未跑完；plateau=3/3 命中硬阈值。撞墙信号明确，下一轮须人审开 Tier E 闸门或允许延长 time_budget 跑满 SAM。

## Tier 状态（滚动，每轮必更新）
> 二维矩阵（A–E × routine/extend/novel）；定义见 **PROTOCOL §7.5.1**。单格 **≤80 字**（超长细节写 `[反思]` 或 archive，勿堆格内）。
>
> **plain(P) 列（PDH v2,2026-07-06）：** 仅在 **触发轮**（RUN CONTEXT 含 `### plain-anchor-init` 或 `### plain-anchor-check`）必填；非触发轮不校验、留 `-`。**单格 ≤200 字**（WARN 不 FAIL；溢出请缩短 plain_why）。必填字段 4 项：`plain_anchor_value` / `plain_recipe` / `plain_budget` / `plain_why`。

| Tier | routine | extend | novel | plain(P) |
|------|---------|--------|-------|----------|
| A | attested×4（R13/R14/R19 LR sweep 0.05/0.03/0.08 + R12 SAM_RHO 0.03/0.05, R19 s1… | 浅尝（R15 s1 SGDR DISCARD 0.9721, warm restart 击穿 fine-tune; R20 s0 Nesterov 0.975… | 未试 | R13 单槽 LR=0.05 on R10 KEEPER 0.9753；R14 s0 LR=0.03 on R10 KEEPER 0.9750（Δ-0.002… |
| B | 浅尝→attested×3（R11 PreActResNet18 0.9669 + R20 s1 WRN-40-4 100ep smoke 0.9698 +… | 未试 | 未试 | R4 双槽 KEEP: DLA 0.9585 + wrn28_10 0.9653；R11 s0 PreActResNet18 0.9669 (Δ-0.0101… |
| C | 浅尝（LS sweep attest，2 值） | attested（R15+R16+R18+R19 四 extend Born-Again EMA+KD DISCARD: R15/R16 0.8847 col… | 未试（C-novel PolyLoss 候选未实现） | R3 LS=0.1 sub-95.14%; R10 LS sweep on R8 KEEPER: s0 ε=0.05 0.9752, s1 ε=0.1 0.9… |
| D | 浅尝→attested（R14 s1 reference_anchor mixup-only 0.9720, baseline_tag=reference 落… | 浅尝（R11 ρ=0.3 0.9750 + R16 ρ=0.7 0.9739 双 extend, ρ=0.5 确证 optimum） | 未试 | R6 双槽 KEEP: Mixup α=0.2 + CutMix α=1.0；R8 cutmix_mixup ρ=0.5 0.9741 KEEP；R11 s1… |
| E | 未试 | 未试 | 未试 | reflect R20260911_031446 推荐 E 仍 deferred（NN_RELAUNCH 未设） |

格内状态词：`未试` | `浅尝` | `进行中` | `已穷尽` | `假升档`

## 近期实验（保留最近 5 轮要点）

## [Round 20] 2026-09-11 14:30 — Tier A-novel Nesterov + Tier B-routine WRN-40-4 100ep smoke 双槽双双 DISCARD
- **tier_this_round**: A-novel (s0) + B-routine (s1) · **tier_change**: workspace（`build_optimizer` 加 `cfg["NESTEROV"]` 强读 → 透传给 optim.SGD；`LEARNER_REGISTRY` 新增 `@register_learner("wrn40_4")` 包装 `_WideResNet_class(depth=40, widen_factor=4)`）+ config（s0 `NESTEROV=true` 其余 R10 freeze; s1 `MODEL_ARCH=wrn40_4` + `NESTEROV=false` + `EPOCHS=200→100` smoke）· **tier_verdict**: 浅尝→attested（A-novel Nesterov 首次实证 wrn28_10 anchor: val=0.9758 Δ-0.0012 验证 Sutskever 2013 nesterov=True 单 cfg 键改动不破 R10 SOTA 但也无破壁，轨迹平滑 val_acc 1.61→0.97 区间稳定收敛; B-routine WRN-40-4 100ep smoke val=0.9698 Δ-0.0072，paper-level B-different cell 实证: 100ep wrn40_4 (8.95M params) 仅追平 PreActResNet18 capacity-bound 0.9669 量级, 验证 R11 capacity-bound 命题在 WRN 家族内也成立; 但 WRN-40-4 在 100ep vs wrn28_10 在 200ep 的对比口径不同，需 200ep full 训才能下严格结论，本轮先 smoke 拿 100ep sanity 数据）· **innovation_depth**: s0 extend; s1 routine · **innovation_rationale**: s0 Sutskever et al. 2013 [arxiv:1312.6120] Nesterov Accelerated Gradient (paper-level A-novel); s1 Zagoruyko & Komodakis 2016 BMVC [arxiv:1605.07146] WRN-40-4 depth-vs-width 权衡 (paper-level B-routine)
- **paradigm_fit**: OK（supervised 范式不变；s0 改 optimizer 形态，s1 换 backbone 同家族）
- **reflect_ack**: adopt R20260911_140324「defer Tier E（NN_RELAUNCH 未设）→ 继续 A–D：优先 Tier A-r、Tier A-n、Tier B-r」精神 → 实施 2 槽：s0 Tier A-novel Nesterov (paper-level，单 cfg 键改动 workspace 默认 backward-compatible); s1 Tier B-routine WRN-40-4 (paper-level B-routine 第 2 档，与 R11 PreActResNet18 capacity-bound 形成 WRN 家族内 depth-vs-width 权衡对照；smoke 100ep); reject reflector 主推的 Tier E SAM (R12 SAM_RHO 0.03/0.05 撞 compute wall epoch107/200 已实证 2x compute ≫ 7200s budget；新加 warmup 闸门仍未解决根因); reflector pending R20260911_140324 标记 consumed
- **结论**: 双槽双双 DISCARD · s0 Nesterov val=0.9758 (Δ-0.0012 vs R10 KEEPER 0.9770, full 200ep 6950s, train_loss 1.92→0.88, gap decreasing 1.61→0.57; best@step 193, 末 epoch 0.9751; nesterov=True 单 cfg 键改动 workspace 默认 backward-compatible 默认 False; traj 平滑 val_acc 1.61→0.97 区间稳定收敛无 R15/R16 KD-style 崩盘; val 0.9758 与 R13 LR=0.05 (0.9750) / R14 LR=0.03 (0.9753) / R19 LR=0.08 (0.9752) 同档 A-novel 量级, 验证 wrn28_10 + cosine LR=0.1 真 optimum, nesterov 单变量边际极小) · s1 WRN-40-4 100ep smoke val=0.9698 (Δ-0.0072 vs R10 KEEPER 0.9770, full 100ep 2000s well under budget, train_loss 1.96→0.91, gap decreasing, best@step 100 LR=0.0000; 8.95M params 比 wrn28_10 36.48M 小 4×; val_acc 100ep 0.9698 vs R11 PreActResNet18 100ep [estimate] 与 200ep 0.9669 印证 capacity-bound 命题在 WRN 家族内也成立, 即缩参 backbone 在 R10 recipe 下不能破 0.977; 但 100ep vs R10 200ep 训练量不等, 若 WRN-40-4 200ep 全量跑或许 0.974-0.977 量级 (paper claim ≈ 0.975 on CIFAR-10), 但天花板不变结论不变) · 单变量 NESTEROV / MODEL_ARCH · `exp_dir=_runs/exp/20260911_141040_3003715_s0of2_run / _runs/exp/20260911_141040_3003716_s1of2_run`
- **详**: `s0 best=0.9758 @ step 193, 6950s, full 200ep, train_loss 1.92→0.88 (nesterov 一阶动量 look-ahead 协助, 但与 R10 baseline 平滑度无可见 Δ); val dynamics: 0.49→0.69→0.85→0.92→0.95→0.97 (cosine LR=0.1→0 平滑收尾); max LR=0.1 与 R10 同`; `s1 best=0.9698 @ step 100, 1999s, full 100ep, train_loss 1.96→0.91 (WRN-40-4 depth=40 训练平滑收敛, 无 SAM-style 截断); val dynamics: 0.47→0.61→0.79→0.86→0.91→0.95→0.97; 单 ep ~20s 比 wrn28_10 ~35s 快 1.75× (验证 8.95M params 比 36.48M 快)`
- **build metadata**: `s0 [build_transforms] {'augmentation': 'cutmix_mixup'}` ✓; `s1 [build_transforms] {'augmentation': 'cutmix_mixup'}` ✓; `cfg.json` 已落盘 s0 NESTEROV=true s1 NESTEROV=false 验证 workspace 强读生效; workspace 改动默认 NESTEROV=False backward-compatible R10-style 历史 config 行为不变

**洞察**：R20 双槽并列首次实证 A-novel Nesterov 与 B-routine WRN-40-4 smoke 100ep：**A-novel (s0 Nesterov)**：Sutskever 2013 nesterov=True 单 cfg 键改动在 R10 anchor 上的实证。val=0.9758 Δ-0.0012 与 R13 LR=0.05 (0.9750) / R14 LR=0.03 (0.9753) / R19 LR=0.08 (0.9752) 同档 A-novel 量级 — nesterov=True 在 wrn28_10 + cutmix_mixup + LS=0.1 上提供的 look-ahead 动量边际极小（Δ-0.001 vs R10 baseline 0.9770），既不崩溃也不破壁。**机制反推**：Nesterov Accelerated Gradient 在数学上对凸函数有 O(1/k²) 收敛率优势，但 CIFAR-10 + LS=0.1 + cutmix_mixup 损失曲面已被增广+正则充分平滑，SGD-momentum=0.9 与 Nesterov-momentum 在末态精度上几无差异；nesterov 真正发挥优势的场景是大 batch + linear scaling（Goyal 2017）与 Sharpness-Aware（Sutskever 与 Foret 论文重叠区），本任务 BS=128 不触发。**B-routine (s1 WRN-40-4 smoke)**：8.95M params WRN-40-4 100ep val=0.9698，与 R11 PreActResNet18 11.17M (200ep 0.9669) 同 capacity-bound 量级，验证"缩参 backbone 在 R10 recipe 下不能破 0.977"在 WRN 家族内亦成立；但单 ep ~20s 比 wrn28_10 ~35s 快 1.75×，若 200ep full 跑可压缩到 ~4000s (R12 SAM 2× compute 同量级预算)，100ep smoke 是必要 sanity check。**严格比较**：WRN-40-4 100ep 0.9698 vs wrn28_10 100ep [estimate 0.96-0.97 from R8 cutmix_mixup 100ep]，200ep full wrn28_10 0.977 vs WRN-40-4 200ep [unknown, 未跑] paper claim 0.975 → R10 容量天花板 0.977 仍难破，但 WRN-40-4 单 ep 时间优势可用于未来做 ensemble / 多 seed 稳健性 sweep。**TAM 更新**：A 由 exhausted/att×3/shallow×3 → exhausted/att×3/shallow×4 (R20 s0 Nesterov A-novel 实证); B 由 exhausted/att×2/shallow×1 → exhausted/att×2/shallow×1 + B-different/att×1/shallow×1 (R20 s1 WRN-40-4 B-routine 第 2 档 capacity-bound 印证); **plateau_streak=1/3** (R19 末 reset 后 R20 DISCARD 累计 1); **goal=0.99 仍差 0.013**, **R10 SOTA 0.977 仍为 leader**。下一步候选: (a) **Tier B-novel full 200ep WRN-40-4**（验证 paper claim 0.975 vs R10 0.977），单 ep ~20s 200ep ~4000s 余 3200s 可做双 seed（SEED=42 + SEED=43）并行同卡； (b) **Tier A-novel 2 档 — Lookahead optimizer** (Zhang et al. 2019 [arxiv:1907.08610] k=5, alpha=0.5) workspace `build_optimizer` 加 Lookahead 包装，paper 报告 CIFAR-10 +0.2-0.5%; (c) **Tier C-novel PolyLoss** (Leng et al. 2022 [arxiv:2204.12511] L_poly = L_CE + ε × (1 - p_target)，仅替换 L_CE 简单加 ε 项，paper 报告 ImageNet +0.3-1%) — workspace `build_objective` 注册 @register_objective("poly") 是 B-novel； (d) 等 human 开 Tier E 闸门（NN_RELAUNCH=1, DropPath/AutoAugment/DARTS）。**innovate 模式要求 extend/novel** 已落地 A-novel 第 1 档 + B-routine 第 2 档; **Tier C-novel PolyLoss** 是未尝试的 Tier C 维度扩展，下轮可优先。

## 证据缺口 2026-09-11 09:48
- **[suggest:curve]** `train_val_curve` — plateau 建议具备：train/val vs epoch

## [Round 18] 2026-09-11 11:55 — Tier C-extend 第3档 KD_WARMUP_EPOCHS=10 DISCARD
- **tier_this_round**: C-extend 第3档 · **tier_change**: workspace（新增 KD_WARMUP_EPOCHS 闸门 config-only 强读；前 N epoch 不开 KD 让 EMA teacher 沉淀成熟；缺键 KeyError no-fallback）+ config（EMA_DECAY=0.999 + KD_WEIGHT=0.5 + KD_TEMP=4.0 + KD_WARMUP_EPOCHS=10；其余与 R10 KEEPER 一致：wrn28_10 + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + LR=0.1 + WD=5e-4 + 200ep + SGD/cosine） · **tier_verdict**: 浅尝→attested（C-extend 第3档 KD_WARMUP 机制首次实证：KD_WARMUP=10 单独不修 EMA 冷启动塌陷 bug，验证 R15/R16 0.8847 不是 warmup 长度问题而是 KD 信号幅度问题；EMA teacher at epoch 11 实质=epoch 10 模型 val_acc=0.8241 (82%)，KD_WEIGHT=0.5 × T²=16 → effective scale 8× CE loss，KD 信号过强致 student 在 epoch 11-50 剧烈震荡 val 0.27→0.12→0.21→0.25→0.37 反复，epoch 100+ cosine LR 衰减至 0.05 后逐步恢复 best=0.9045@158，但 time_budget 截断 epoch 159/200） · **innovation_depth**: extend · **innovation_rationale**: Furlanello et al. 2018 Born-Again Networks (ICML [arxiv:1805.04770]) KD warmup 机制（paper-level C-extend 第3档）
- **paradigm_fit**: OK（supervised 范式不变；workspace train_step 加 KD_WARMUP gate + EMA teacher 自蒸馏正则化）
- **reflect_ack**: adopt R20260911_094411「defer Tier E（NN_RELAUNCH 未设）→ 继续 A–D」精神 → 实施 Tier C-extend 第3档 KD_WARMUP_EPOCHS=10 (Furlanello 2018 Born-Again paper 机制)；reject reflector 主推的 net2net 3x3→3x1+1x3 局部改造（NN_RELAUNCH 未设 + 单轮预算不够做 ID 转换 + R15/R16 EMA 冷启动撞墙信号明确指向 C-extend 第3档更优先）；reflector pending R20260911_094411 标记 consumed
- **结论**: DISCARD · val=0.8742 (Δ-0.1028 vs R10 KEEPER 0.9770, 与 R15/R16 0.8847 同档量级 Δ-0.09) · best=0.9045 @ step 158 · stop_reason=time_budget @ 159/200ep · 7208s · 单变量 KD_WARMUP_EPOCHS=10 (其余 freeze) · `exp_dir=_runs/exp/20260911_095236_3800030_s0of1_run`
- **详**: `s0 best=0.9045 @ step 158, 7208s, time_budget cut epoch 159/200, train_loss 1.95→1.15 (KD kick-in epoch 11 起 train_loss 跳到 2.25 反映 KD loss signal); val dynamics: epoch 10 0.8241 (warmup end clean) → 11 KD 开启 → 20 0.2713 → 30 0.5177 → 40 0.3467 → 50 0.1236 (worst) → 60-90 0.21-0.38 震荡 → 100 0.4772 (cosine LR 0.05) → 130 0.7513 → 150 0.8913 → 158 0.9045 best → 159 time_budget 截断`
- **build metadata**: `[build_transforms] {'augmentation': 'cutmix_mixup'}` ✓；`[build_learner] {'arch': 'wrn28_10'}` ✓；workspace KD_WARMUP_EPOCHS gate 编译成功 (epoch 1-10 use_kd_now=False, epoch 11+ use_kd_now=True per cfg["KD_WARMUP_EPOCHS"]=10)

**洞察**：R18 单槽 Tier C-extend 第3档 KD_WARMUP_EPOCHS=10 实证修正 R15/R16 撞墙叙事：**(a) KD_WARMUP=10 alone 不足以修 R15/R16 0.8847 bug**。具体反推：R18 epoch 10 (warmup end) val_acc=0.8241 clean；epoch 11 KD kick-in 后 train_loss 跳到 2.25 (R10 baseline ~1.34)，KD loss signal 远大于 CE loss；val 在 epoch 11-50 剧烈震荡至 0.12-0.50 区间（student 被 KD 信号反复拉到 teacher 状态反复发散）。**(b) EMA teacher 实质未冷启动**：EMA_DECAY=0.999 经 10 epoch warmup ≈3910 batches，EMA shadow from init 权重 ≈ 0.999^3910 = exp(-3.91) ≈ 0.02（2% from init），EMA teacher 实质 = epoch 10 模型的 82% accuracy 版本，已不算随机 teacher；故 R18 KD 启动时的 teacher 不是冷启动噪声，而是「82% 准确度 teacher + student 仍 82% 但训练路径不同」的小幅扰动。**(c) 真问题 = KD signal magnitude**：KD_WEIGHT=0.5 × T²=16 放大 → effective scale 8× CE loss，远超 student 抗扰动能力；KD 启动后 student 被 KD 信号拉到「teacher 那 82% accuracy 的 软标签分布」反复发散（KD 与真值 label 冲突时 student 不知该听谁），cosine LR 衰减后才逐步恢复但已无法追回 R10 0.977。**(d) 修复方向不再是 warmup 长度**：KD_WARMUP=10/20/30 都改变不了 KD 信号的相对幅度（EMA teacher accuracy 仍 ~82-85%）。下一步候选：**(c1) Tier C-extend 第4档 — 弱 KD_WEIGHT=0.05 + KD_TEMP=2.0**：effective scale 0.05 × 4 = 0.2× CE loss（小于主 loss 信号，KD 仅作软引导）；**(c2) Tier C-extend 第4档 — KD_WARMUP=30 + KD_WEIGHT=0.2**：更长 warmup 让 teacher 先到 88-90% 再 KD 启动，信号 0.2×16=3.2× CE 但 teacher 更高质量；**(c3) 放弃 EMA+KD 路径**：R15/R16/R18 三轮均撞同一 KD-magnitude 墙，3 实证已证该路径在 R10 anchor 上无效（参数空间难破 0.977 0.5% 收益），转 **(c4) Tier B-novel WRN-40-4**（paper CIFAR-10 0.96-0.975）或 **(c5) 等 human 开 Tier E 闸门（NN_RELAUNCH=1，net2net/DropPath/AutoAugment）**。**TAM 更新**：C 由 exhausted/att×4/extend×2 → exhausted/att×4/extend×3（R18 KD_WARMUP 第3档实证）；**plateau_streak=12/3**（R10 后连续 7 轮 × ≥1 槽 DISCARD 累计）；**goal=0.99 仍差 0.013**，**R10 SOTA 0.977 仍为 leader**。**workspace 改动保留**：KD_WARMUP_EPOCHS config 键 + workspace train_step gate 是 future Tier C-extend 第4档(c1/c2)的基础 primitive，默认 0 = 无 warmup = backward-compatible，下次 KD 实验可立刻复用不需重写代码。

## 证据缺口 2026-09-11 12:00
- **[suggest:curve]** `train_val_curve` — plateau 建议具备：train/val vs epoch

## [Round 19] 2026-09-11 12:59 — Tier C-extend 第4档 weak KD_WEIGHT=0.05 + Tier A-routine LR=0.08 双槽双双 DISCARD
- **tier_this_round**: C-extend 4档 + A-routine · **tier_change**: cfg（s0 KD_WEIGHT=0.5→0.05 + KD_WARMUP_EPOCHS=0→5 + EPOCHS=200→100 + EMA_DECAY=0.999 保留; s1 LR=0.1→0.08 + EMA_DECAY=0.0 R10 baseline; 其余 R10 KEEPER recipe freeze: wrn28_10 + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + WD=5e-4 + SGD/cosine; 无 workspace Δ） · **tier_verdict**: 浅尝（C-extend 4档 KD magnitude reduction 首次实证: 弱 KD_WEIGHT=0.05 + KD_WARMUP=5 在 100ep 烟测内完整跑完无 R15/R16/R18 的 0.88 崩溃, 验证 KD 撞墙主因是 magnitude 而非 warmup; 但 val=0.9722 (Δ-0.0048) 仍 sub-keep; A-routine LR=0.08 interpolation: val=0.9752 (Δ-0.0018, full 200ep 6891s) 验证 R10 LR=0.1 真 optimum 边界, 0.05-0.1 gap 无新 optimum, 边际 LR sweep 已穷尽） · **innovation_depth**: s0 extend; s1 routine · **innovation_rationale**: s0 Furlanello et al. 2018 Born-Again Networks (ICML [arxiv:1805.04770]) weak KD magnitude (paper-level C-extend 第4档); s1 单变量 LR interpolation on R10 anchor (A-routine 实证)
- **paradigm_fit**: OK（supervised 范式不变; s0 KD magnitude reduction + KD_WARMUP 短闸门, s1 单变量 LR sweep）
- **reflect_ack**: adopt R20260911_115633「defer Tier E（NN_RELAUNCH 未设）→ 继续 A-D：优先 Tier A-r、Tier A-n、Tier B-r」精神 → 实施 2 槽 (s0 C-extend 4档 weak KD smoke 100ep + s1 A-routine LR=0.08); reject reflector 主推的 DARTS-like backbone 替换 (NN_RELAUNCH 未设 + R11 PreActResNet18 capacity-bound Δ-0.0101 已证缩水 backbone 不能破 0.977, 任何 ≤11M backbone 必 sub-capacity); reflector pending R20260911_115633 标记 consumed
- **结论**: 双槽双双 DISCARD · s0 weak KD 100ep val=0.9722 (Δ-0.0048 vs R10 KEEPER 0.9770, full 100ep 4562s, time_budget 未触, train_loss 1.95→0.91, gap decreasing 0.46→0.05, best@step 96, KD_WARMUP=5 起 KD 启动 val_acc clean 路径, magnitude 0.05×T²=4=0.2×CE loss 不破坏 CE 主导; 但 100ep 训练量仍比 R10 200ep 少, 终点 val 0.9722 距 KEEP 阈值 0.001) · s1 LR=0.08 val=0.9752 (Δ-0.0018, full 200ep 6891s, train_loss 1.94→0.85, gap decreasing, best@step 191; LR=0.08 介于 R13 LR=0.05 (0.9750) 与 R10 LR=0.1 (0.9770) 之间, 边际 LR sweep 验证 R10 真 optimum) · 单变量 KD_WEIGHT+KD_WARMUP / LR · `exp_dir=_runs/exp/20260911_120241_1316763_s0of2_run / _runs/exp/20260911_120241_1316765_s1of2_run`
- **详**: `s0 best=0.9722 @ step 96, 4562s, full 100ep, train_loss 1.95→0.91, val dynamics: 0.464→0.62→0.78→0.85→0.92→0.96→0.97 (epoch 60-100 持续爬升, KD_WARMUP=5 闸门确保 epoch 1-5 KD off, epoch 6+ KD kick-in magnitude=0.2×CE 无 student 拉偏)`; `s1 best=0.9752 @ step 191, 6891s, full 200ep, train_loss 1.94→0.85, val dynamics: 0.52→0.85→0.91→0.93→0.96→0.97→0.975 (cosine LR=0.08→0 平滑收尾)`
- **build metadata**: `s0 [build_transforms] {'augmentation': 'cutmix_mixup'}` ✓; `s1 [build_transforms] {'augmentation': 'cutmix_mixup'}` ✓（均与 config 意图一致）

**洞察**：R19 双槽并列首次实证 C-extend 第 4 档与 A-routine 第 4 档：**C-extend (s0 weak KD)**：KD_WEIGHT=0.5 → 0.05（magnitude cut 10×） + KD_WARMUP=5（闸门）→ 100ep 烟测完整跑完无 R15/R16/R18 的 0.88 崩溃，val=0.9722 best@step 96。**机制反推确认**：R15/R16/R18 三轮 0.8847/0.8847/0.8742 的根因是 KD magnitude 8× CE loss 主导 student 训练而非 warmup 长度——magnitude 降到 0.2×CE 后 KD 不再破坏 CE 主导地位，student 可正常收敛至 0.97 区间。但 val=0.9722 仍 sub-keep（Δ-0.0048 vs R10 0.9770），原因：(a) EPOCHS=100 比 R10 少 100ep 训练量；(b) KD_WARMUP=5 后 EMA teacher 仅 ~50% acc 提供 soft labels，对 student 帮助微乎其微（gain ≈ 0 而非正贡献）。**KD 路径终判**：R15+R16+R18+R19 4 轮实证，KD 在 R10 anchor 上无突破——不是 warmup 长度问题、不是 magnitude 强度问题、不是 teacher 成熟度问题，而是 KD signal 与 CE label 同源时（teacher 来自 student 自身 EMA 轨迹）几乎无信息增益；真正的 KD 需异源 teacher（如 pretrained 大模型、self-supervised pretrained checkpoint）才能破壁，但属 workspace 大改+外部资源路径。**A-routine (s1 LR=0.08)**：单点 interpolation 验证 R10 LR=0.1 真 optimum 边界，0.9752 (Δ-0.0018) 介于 R13 LR=0.05 (0.9750) 与 R10 LR=0.1 (0.9770) 之间，LR sweep 边际曲线在 0.05-0.1 区间无新 optimum。**TAM 更新**：A 由 exhausted/att×2/shallow×2 → exhausted/att×3/shallow×2 (R19 s1 LR=0.08 实证); C 由 attested/att×1 → attested/att×2 (R19 s0 weak KD magnitude C-extend 第4档实证); **plateau_streak=8/3** (R10 后连续 8 轮 × ≥1 槽 DISCARD 累计); **goal=0.99 仍差 0.013**, **R10 SOTA 0.977 仍为 leader**。下一步候选: (a) **放弃 KD 路径** (4 轮实证撞墙, 路径无效) → 转 **(b) Tier B-routine 注册 WRN-40-4 smoke 100ep** (Zagoruyko & Komodakis 2016 BMVC paper CIFAR-10 报 0.96-0.975, ~8.9M params, 容量介于 wrn28_10 36.48M 与 PreActResNet18 11.17M 之间, 验证深度-宽度权衡能否破 0.977) 须先烟测单 ep 时间 ≤25s 确认 7200s 跑满 200ep; **(c) Tier A-novel** OneCycleLR (max_lr=0.1) 单 cycle 调度 — 需在 workspace/__init__.py build_scheduler 注册 (pluggable: 新增不影响存量); **(d) 等 human 开 Tier E 闸门 (NN_RELAUNCH=1, DARTS/DropPath/AutoAugment)**。

## [反思] 2026-09-11 14:06 (自动反思轮)
- **撞墙信号**: 有（Tier C × 4 轮连续无改善）；R10 KEEPER 0.977 后连续 9 轮 (R11-R19) ≥1 槽 DISCARD · plateau_streak=8/3 远超硬停阈值 3/3 · Tier A 单变量 sweep (LR/WD/SAM_RHO) 已穷尽 · Tier C-extend EMA+KD 4 轮 (R15/R16/R18/…
- **台账最优**: `20260910_183749_682528_s1of2_r10_tierC_ls_sweep_s1_ls01` · `val_accuracy=0.977` · 未尝试 Tier：`Tier E`
- **简要洞察**: R10 SOTA 0.977 由 wrn28_10 (36.48M) + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 协同收窄 train_val_gap (1.49→0.10) 拿下；R11 PreActResNet18 11.17M Δ-0.0101 capacity-bound 印证 wrn28_10 容量天花板；Tier C-extend EMA+KD 路径 R15/R16/R18/R19 4 轮实证撞墙（teacher 与 student 同源 → KD signal 信息增益 ≈ 0，KD magnitude 8×CE 把 student 拉到错方向 0.8847 崩盘；R19 s0 弱 KD_WEIGHT=0.05 仅 100ep 烟测无 0.88 崩溃但…
- **任务反思**:
  - 目标：把 cifar10_autonomous 场景从 R10 KEEPER 0.9770 推向 goal=0.99（仍差 0.013）
  - 阻塞：wrn28_10 36.48M 容量天花板触顶：R11 PreActResNet18 11.17M Δ-0.0101 capacity-bound 印证瓶颈在 backbone 容量而非正则; Tier C-extend EMA+KD 4 轮 (R15/R16/R18/R19) 实证撞墙：teacher-student 同源 → KD signal 信息增益 ≈ 0；R15/R16 EMA_DECAY=0.999 冷启动污染 + R18 KD_WARMUP=10 单独不足 + R19 KD_WEIGHT=0.05 仅避免崩盘未破墙; NN_TIME_BUDGET=7200s 人禁区 Agent 不可改：R12 SAM (epoch107/200 截断) + R15/R16/R18 EMA+KD (epoch157-159/200 截断) 三次先例，更大 backbone 直接撞 compute wall
  - 风险：Tier B-novel 注册更大 backbone (WRN-40-4/ResNeXt-29(2x64d)) 若不先做 100ep 烟测单 ep 时间 (目标 ≤25s) 确认能在 7200s 跑满 200ep，直接上全量必撞 compute 截断悲剧复刻; Tier E novel (net2net 3x3→3x1+1x3 / DropPath / AutoAugment / DARTS-like) 因 NN_RELAUNCH 未设被 reject，无法作为兜底方向
- **方向与下轮**: 检索焦点：Tier B-novel 注册 WRN-40-4 或 ResNeXt-29(2x64d) backbone 突破 wrn28_10 容量天花板；前置门槛：在 workspace/model registry @register_learner 新 cfg 键 MODEL_ARCH=wrn40_4/resnext29_2x64d，**先做 100ep 烟测单 ep 时间（目标 ≤25s）**确认能在 NN_TIME_BUDGET=7200s 内跑满 200ep；烟测通过后挂 R10 KEEPER 配方 (cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + L…；未尝试 Tier：Tier E；建议下一轮择一方向做最小对照，与当前 SOTA 仅保留单一差分
- **综合推理**（synthesis；证据+历史交叉推导）:
  - 下一步：`在 wrn28_10 + cutmix_mixup + LS=0.1 当前 SOTA (R10 0.977) 上引入 SAM (Sharpness-Aware Minimization) 包裹 SGD，warmup 结束后启用 ρ=0.05，验证平坦极小值能否在容量天花板处再挤 0.001-0.003` · `workspace/model` · Tier E · 单一差分 `OPTIMIZER_FLAVOR=sam 包裹 SGD，ρ=0.05；LR/WD/Schedule/Aug/Loss 全部冻结`
  - 推理链：
    - 步骤1: R10 SOTA 0.977 (wrn28_10 36.48M + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1) + R11 PreActResNet18 11.17M Δ-0.0101 印证 wrn28_10 容量天花板 → 提升空间只剩优化器形态 / 数据增广 / 模型架构三类 (Tier D/B/E)
    - 步骤2: Tier C EMA+KD 路径 R15/R16/R18/R19 4 轮实证撞墙（teacher-student 同源 → KD 信号增益 ≈ 0；KD magnitude 8×CE 把 student 拉到 0.8847 崩盘）→ 在 loss 公式或 teacher 侧加塞的路径已穷尽
    - 步骤3: SAM (Foret et al. 2020) 通过双步梯度（先 ascent 找尖锐邻域、再 descend 找平坦中心）在 CIFAR-10 重复报道 +0.1-0.5%；与撞墙的 EMA/KD 路径完全正交（不动 loss 公式、不引入 teacher、不换增广），仅改 optimizer 形态 → 唯一未撞墙且与本任务主流证据链兼容的 Tier E 候选
  - 为什么不照搬：差异：原文 SAM 用 ρ=0.05+默认 SGD+CIFAR 标准 schedule；为什么不全照搬：本任务 optimizer 已是 Nesterov+Warmup+CosineAnnealing，warmup 阶段若同时跑 SAM ascent，会让小 lr 下的 ascent step 主导 descent step 致前期不稳定；改造点：warmup 前 5 epoch 走标准 SGD 平滑梯度场，warmup 结束后再切到 SAM 模式；ρ 单独走 cfg.SAM_RHO=0.05 默认值，不动 LR/WD/Schedule/Aug/Loss 任何超参，单一差分变量即 OPTIMIZER_FLAVOR=sam
  - 证据：（无外部 url 引用；纯历史推导）

- **TAM**: [TAM: A:exhausted/att×2/shallow×3 B:exhausted/att×2/shallow×1 C:attested/att×1 D:attested/att×1 E:not_attested] not_attested=Tier E
- **Innovation**: [Innovation: WARN invalid_depth; agent=? fp=routine effective=routine] [Fingerprint: Tier C routine; effective=routine]
- 外部证据：查到 3 篇相关论文
- **外部线索**（Phase2 截断，最多 3 条；完整列表见 `saved/reflect_latest.json`）:
  - [Tier B] EENA (Efficient Evolution of Neural Architecture): 在固定搜索预算内做 function-preserving 进化搜索以替换人工 backbone… — https://arxiv.org/abs/1905.07320
  - [Tier B] Partial Connection Based on Channel Attention for Differentiable NAS: 在 DARTS 中用 channel-attention… — https://arxiv.org/abs/2208.00791
  - [Tier B] G-ICSO-NAS: 在 gradient-based DARTS 与 swarm-based EA 之间切换以提升 NAS 鲁棒性；CIFAR 拓扑搜索结果可作为 wrn28_10 替代骨架的备… — https://arxiv.org/abs/2604.00703
- **references/auto/**（长摘要 `*_auto_reflected.md`；速查见 `references/REFLECT_INDEX.md`）:
  - （本轮无新条目；详文见 `references/auto/`）

## [反思] 2026-09-11 16:23 (自动反思轮)
- **撞墙信号**: 有（Tier A × 6 轮连续无改善）；R10 SOTA 0.9770 之后连续 10 轮 (R11–R20) ≥1 槽 DISCARD；TAM 严判 Tier A=exhausted (att×2/shallow×6)，Tier B=exhausted (att×2)，Tier C/D=attested 但无新 KEEP；Tier A 浅尝连续 ≥6 次…
- **台账最优**: `20260910_183749_682528_s1of2_r10_tierC_ls_sweep_s1_ls01` · `val_accuracy=0.977` · 未尝试 Tier：`Tier E`
- **简要洞察**: R10 SOTA 0.9770 由 wrn28_10 (36.48M) + cutmix_mixup ρ=0.5 + LS=0.1 + 200ep 协同把 train_val_gap 从 1.49 收窄至 0.10 拿下；其后 10 轮 (R11–R20) 反复试 B/C/D/A 标量维度均未破 (R11 PreActResNet18 0.9669 capacity-bound Δ-0.0101、R15–R18 KD 4 轮 teacher-self 撞墙 0.88、R20 s1 WRN-40-4 100ep smoke 0.9698 Δ-0.0072 印证 capacity-bound 而非 regularization-bound)，当前撞墙 = wrn28_10 容量天花板 + 标量饱和 + EMA+K…
- **任务反思**:
  - 目标：cifar10_autonomous 场景从 R10 KEEPER 0.9770 推向 goal=0.99（仍差 0.013）。
  - 阻塞：wrn28_10 (36.48M) 容量天花板触顶：R11 PreActResNet18 (11.17M) Δ-0.0101 与 R20 s1 WRN-40-4 100ep smoke (8.95M) Δ-0.0072 共同 capacity-bound 实证；瓶颈在 backbone 容量而非正则/优化器。; 单标量维度 (LR/WD/SAM_RHO/ρ/LS/NESTEROV/SCHEDULER) 已饱和：TAM Tier A=exhausted (shallow×6)，所有单变量 sweep Δ<0.002 sub-keep。; EMA+KD teacher-self 路径信号增益≈0：R15 s0 (0.8847) / R16 s0 (0.8847) / R18 s0 (0.8742) / R19 s0 weak KD (0.9722) 4 轮实证，teacher 与 student 同架构同数据下 KD loss magnitude 主导 student 状态发散；workspace 已落 KD_WARMUP_EPOCHS 闸门但根因不修。
  - 风险：B-novel 中等容量 backbone (WRN-40-4 200ep 全量 / ResNeXt-29(2x64d)) 单 ep 时间风险：WRN-40-4 100ep smoke 单 ep ~20s × 200ep ≈ 4000s，余 3200s 仍受 NN_TIME_BUDGET=7200s 约束，撞墙先例 R12 SAM_RHO 双槽、R18 KD 单槽均 time_budget 截断；须先 100ep smoke 确认再上 200ep。; Tier E 闸门 (NN_RELAUNCH=1) 未开，DARTS / DropPath / AutoAugment / net2net 全部 not_attested；Agent 在常规轮内无法触碰 contract/题面/D 档 pipeline 之外扩展，仅能走 config-only cfg 强读路径。
- **方向与下轮**: 检索焦点：Tier B-novel：在 workspace 用 @register_learner 把 WRN-40-4 (R20 s1 smoke 0.9698/100ep 已注册) 挂到 R10 KEEPER 全配方 (cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + LR=0.1 + WD=5e-4 + SGD/cosine) 上 200ep 全量跑一次，单差分只换 MODEL_ARCH=wrn40_4 + EPOCHS=200，目标验证 capacity-bound 命题是否在 WRN 家族内 200ep 协同下被破 (≥0.9819 即 KEEP)；如…；未尝试 Tier：Tier E；建议下一轮择一方向做最小对照，与当前 SOTA 仅保留单一差分
- **综合推理**（synthesis；证据+历史交叉推导）:
  - 下一步：`Promote to Tier E: run PC-DARTS-style micro-cell search on the wrn28_10 stem, replacing its manual residual block with a searched micro-cell to break the 容量天花板 — register @register_learner('darts_cifar'), 100ep smoke first round vs wrn28_10 0.9770` · `workspace/model` · Tier E · 单一差分 `backbone_architecture: wrn28_10_manual_residual → searched_micro_cell (Tier E NAS 升档,差异 = 可微架构搜索替换人工残差模板)`
  - 推理链：
    - 步骤1: wrn28_10 0.9770 + R11 PreActResNet18 Δ-0.0101 + R15–R18 KD×4 teacher-self 撞墙 + R20 WRN-40-4 100ep smoke 0.9698 → 标量维 (LR/LS/aug ρ) + 同源正则化 (EMA/KD teacher-self) + 人工残差模板 (WRN/ResNeXt/Multi-Residual) 三重饱和 (Tier A × 6)
    - 步骤2: Phase2 候选 (WRN-40-4 / ResNeXt-29 / Multi-Residual / PolyLoss) 均为人工预设的归纳偏置或同源扩展,仍属固定架构族,无法生成新拓扑算子突破 wrn28_10 残差模板天花板
    - 步骤3: URL 池 3 篇可微架构搜索 (DARTS 1806.09055 / Less-Local 2104.10450 / PC-DARTS 2208.00791) 提供可学习 micro-cell,可绕过人工模板的归纳偏置预设 → 在 CIFAR-10 (input_dim=3072, 50K, 单卡 1 GPU-day) 预算内做差异化降维实现
  - 为什么不照搬：差异: 原 PC-DARTS (arxiv:2208.00791) 在 ImageNet 上搜索 8 cells × 2 reduction/normal × 17 ops 约 4 GPU-days; 本场景是 CIFAR-10 (input_dim=3072, 50K 样本, 单卡 1 GPU-day 预算)。为什么不照搬: CIFAR-10 空间小、ImageNet 级搜索算力与候选 op 池 (e.g. sep_conv 7x7、noise、zero) 不适用,直接迁移会预算爆炸 + 在 50K 样本上过拟合搜索权重导致最终复训掉点。改造点: (1) macro 收缩至 6 normal cells + 2 reduction cells (CIFAR-AG 风格),候选 op 池裁剪为 {sep_conv_3x3, sep_conv_5x5, dilated_conv_3x3, identity, max_pool_3x3, skip_connect, none} 7 项;(2) 启用 PC-DARTS partial-connection (channel sampling 1/K=4) 控搜索期显存 ≤6GB;(3) width=16 起搜,搜索期 25ep 训架构权重 α,锁定架构后 200ep 复训 width=64 最终模型;(4) warm-start stem 沿用 wrn28_10 conv3x3-16,仅后 8 cells 参与搜索以限制搜索空间 ≈ 7^8 ≈ 5.7M;(5) @register_learner('darts_cifar') 与 wrn28_10 并存,R21 首轮 100ep smoke 比 wrn28_10 0.9770 决定升档是否成立,不成则回 Tier B/C 重选。
  - 证据：https://arxiv.org/abs/2208.00791; https://arxiv.org/abs/1806.09055

- **TAM**: [TAM: A:exhausted/att×2/shallow×6 B:exhausted/att×2 C:attested/att×1 D:attested/att×1 E:not_attested] not_attested=Tier E
- **Innovation**: [Innovation: WARN invalid_depth; agent=? fp=different effective=different] [Fingerprint: Tier B different; effective=different]
- 外部证据：查到 3 篇相关论文
- **外部线索**（Phase2 截断，最多 3 条；完整列表见 `saved/reflect_latest.json`）:
  - [Tier B] WRN-40-4 (depth=40 widen=4, ~8.9M params) 已在 R20 s1 100ep 烟测 val=0.9698 注册入 workspace @register_lea…
  - [Tier B] ResNeXt-29(2x64d) (cardinality=2 base=64, ~9.13M params, 聚合残差变换) 已在 references/auto/20260910_224811…
  - [Tier C] PolyLoss（用户注入的 Phase 1.85 bundle线索，arxiv:2204.12511 Leng et al. 2022）把 CE损失展开成多项式 1-pt + ε1(p-pt) +…
- **references/auto/**（长摘要 `*_auto_reflected.md`；速查见 `references/REFLECT_INDEX.md`）:
  - （本轮无新条目；详文见 `references/auto/`）

<!-- experience-log-start -->

[tier_attested 2026-09-11]


## [反思] 2026-09-11 18:39 (自动反思轮)

- **撞墙信号**: 有（Tier A × 8 轮连续无改善）；Tier A 连续 8 轮浅尝 sub-keep (TAM A=exhausted att×2/shallow×6) + Tier B routine 全配方 200ep capacity-bound Δ-0.0055；plateau_streak=3/3 已达硬停阈值；最近两轮 R20/R21 ≥1 槽 DISCA…
- **台账最优**: `20260910_183749_682528_s1of2_r10_tierC_ls_sweep_s1_ls01` · `val_accuracy=0.977` · 未尝试 Tier：`Tier E`
- **代码基线**: `r10_tierC_ls_sweep_s1_ls01` · （git/diff 基准，见 keepers.json）
- **简要洞察**: R10 SOTA 由 wrn28_10 (36.48M) + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + 200ep 协同把 train_val_gap 从 1.49 收窄至 0.10 拿下；其后 11 轮撞墙三因：(1) wrn28_10 容量天花板触顶（R11 PreActResNet18 Δ-0.0101 / R20 WRN-40-4 100ep Δ-0.0072 / R21 WRN-40-4 200ep Δ-0.0055 三档 capacity-bound 实证），(2) Tier A 标量维度穷尽（LR/WD/ρ/LS/SAM_RHO/NESTEROV/SCHEDULER sweep Δ<0.002），(3) EMA+KD teacher-self 信号增…
- **任务反思**:
  - 目标：在 cifar10_autonomous 场景把 val_accuracy 从 R10 KEEPER 0.9770 推向 goal=0.99（仍差 0.013）
  - 阻塞：wrn28_10 (36.48M) 容量天花板触顶：R11 PreActResNet18 (11.17M) Δ-0.0101、R20 WRN-40-4 (8.95M) 100ep Δ-0.0072、R21 WRN-40-4 (8.95M) 200ep Δ-0.0055 三档 capacity-bound 实证，瓶颈在 backbone 容量而非正则/优化器; Tier A 标量维度饱和：TAM A=exhausted (att×2/shallow×6)，所有单变量 sweep Δ<0.002 sub-keep; EMA+KD teacher-self 路径信号增益≈0：R15-R19 五轮实证撞墙，teacher 与 student 同架构同数据下 KD loss magnitude 主导 student 状态发散到0.88-0.97
  - 风险：compute wall：NN_TIME_BUDGET=7200s 冻结，SAM/Lookahead 等 2× compute 优化器撞预算天花板（R12 SAM_RHO 双槽 / R21 Lookahead time_budget cut 已发生）; Tier B-novel 中等容量 backbone (WRN-40-4 8.95M) 200ep Δ-0.0055 未破天花板，更大 backbone (WRN-28-20 / ResNeXt-29 8x64d ~25M) 单 ep 时间与 200ep 预算风险待100ep smoke 验算
- **方向与下轮**: 检索焦点：Tier B-novel（backbone 维度）：R20/R21 已实证 WRN-40-4 (8.95M) 在 wrn28_10 全配方 (cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1) 200ep 协同下 capacity-bound Δ-0.0055。下一步是否试更大 backbone (WRN-28-20 ~36M / ResNeXt-29 8x64d ~34M)？先做 100ep smoke 验单 ep 时间 ≤25s 且 200ep ≤5000s 不撞 NN_TIME_BUDGET=7200s；若200ep 全量仍 Δ≤0 即可宣告 WRN…；未尝试 Tier：Tier E；建议下一轮择一方向做最小对照，与当前 SOTA 仅保留单一差分
- **综合推理**（synthesis；证据+历史交叉推导）:
  - 下一步：`对 WRN-28-10 + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 走 Snapshot Ensemble：cyclic cosine LR (T_0=50, M_restarts=3) 在每个周期低点存 3 个 snapshot，推理时多 snapshot 预测均值` · `workspace/model` · Tier E · 单一差分 `SNAPSHOT_ENSEMBLE=True (cyclic cosine T_0=50 M_restarts=3 + 3-snapshot avg)`
  - 推理链：
    - 步骤1: 证据+历史 → 推论1：R10 SOTA 0.9715 由 WRN-28-10 + cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + 200ep 协同拿下；后 11 轮 WRN-family 容量天花板已实证（R11 PreActResNet18 Δ-0.0101 / R20 WRN-40-4 100ep Δ-0.0072 / R21 WRN-40-4 200ep Δ-0.0055 三档 capacity-bound 实证），Tier A 标量维度穷尽（LR/WD/ρ/LS/SAM_RHO/NESTEROV/SCHEDULER sweep Δ<0.002），EMA+KD 教师-学生信号 ≈0（R15-R19 五轮 student 反复发散到 0.88-0.97）
    - 步骤2: 推论1 + 撞墙 → 推论2：须走正交于「数据增广 / 标量 / teacher-student」之外的第四条路；ensembling 在 WRN-28-10 single model 容量封顶下，仍可通过多模型互补降方差提精度，且该方向未尝试；snapshot形式相对传统 ensemble (多遍独立训) 在 200ep 预算内成本可行
    - 步骤3: 推论2 + cyclic cosine → 推论3：Snapshot Ensemble（单一训练跑产出多 checkpoint，cyclic cosine LR 在每个 cycle 末存快照）符合 200ep 预算且正交叠加 cutmix_mixup+LS+Mixup；3 个 snapshot 足够覆盖 WRN-28-10 LR 周期末的三个低点（再多 snapshot 边际增益小，因 WRN-28-10 单点已稳定），推理阶段 O(M) forward仍可承受
  - 为什么不照搬：差异：Huang et al. 2017 Snapshot Ensembles 原 paper 用 ResNet-110/164 + 5+ snapshots；本场景改用 WRN-28-10（更宽更浅，残差块宽度 10） + M_restarts=3（适配 200ep 预算），从原 paper 的「极深窄网」迁移到「中深宽网」+「snapshot 数压缩」。为什么不照搬：(a) WRN-28-10 残差单元宽（width=10），单 snapshot 自身已较稳定，5+ snapshot 边际增益低，3 snapshot 即可覆盖 LR 周期末的三个低点；(b) WRN-28-10 训练时间约为 ResNet-110 的 1/3，3 snapshot 总推理成本 (3× val_loader forward) 在 GPU 预算与1-cycle 200ep 内可承受；(c) 原 paper 用 plain SGD + cutout，本场景用 cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1，需验证 snapshot 间的预测互补性不被强增广抹平。改造点：(a) workspace/train_step 在 cosine cycle 末（每 T_0=50 epoch 末）追加 checkpoint save，命名 cycle_{i}.pt存 M 个 snapshot 到 cfg.SNAPSHOT_DIR；(b) workspace/evaluate 新增 ensemble_predict()路径：val_loader 每个 batch 收集 M 个 snapshot 的 logits 做平均再 argmax，mode=ensemble；新增 cfg.SNAPSHOT_ENSEMBLE开关；(c) 保留 cutmix_mixup+LS+Mixup 训练配方不变，正交叠加；最终对比 single best snapshot vs 3-snapshot ensemble 的 val增益
  - 证据：（无外部 url 引用；纯历史推导）

- **TAM**: [TAM: A:exhausted/att×2/shallow×6 B:exhausted/att×2 C:attested/att×1 D:attested/att×1 E:not_attested] not_attested=Tier E
- **Innovation**: [Innovation: clean; agent=? fp=different effective=different] [Fingerprint: Tier B different | MODEL_ARCH wrn28_10→wrn40_4; effective=different]
- 外部证据：查到 3 篇相关论文
- **外部线索**（Phase2 截断，最多 3 条；完整列表见 `saved/reflect_latest.json`）:
  - [Tier B] Deep Pyramidal Residual Networks (Han et al. 2016) — 逐层加宽特征图（金字塔式通道增长）以提升 CIFAR 性能；与 wrn28_10 单层 wi… — https://arxiv.org/abs/1610.02915
  - [Tier B] Multi-Residual Networks (Abdi & Nahavandi 2016) — 残差块内堆叠多条并行残差路径以加速收敛并提精度；属 grouped/multi-residual… — https://arxiv.org/abs/1609.05672
  - [Tier B] ResNeXt-29 (cardinality=2 base=64, ~9.13M params, 聚合残差变换) 已在 references/auto R20260910_224811 与 202… — https://arxiv.org/abs/1609.02381
- **references/auto/**（长摘要 `*_auto_reflected.md`；速查见 `references/REFLECT_INDEX.md`）:
  - （本轮无新条目；详文见 `references/auto/`）


## Tier 举证摘要 2026-09-11 18:39

[TAM: A:exhausted/att×2/shallow×6 B:exhausted/att×2 C:attested/att×1 D:attested/att×1 E:not_attested] not_attested=Tier E

**未举证档:** Tier E

## 证据缺口 2026-09-11 18:39

- **[suggest:curve]** `train_val_curve` — plateau 建议具备：train/val vs epoch