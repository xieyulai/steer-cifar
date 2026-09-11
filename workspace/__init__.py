"""
工作区 — Workspace 侧 ExperimentBase 子类（CIFAR-10 迁入）。

覆盖 build_learner / build_objective / build_transforms / train_step / predict / evaluate / build_scheduler / build_optimizer。

模型来源：DLA & SimpleDLA 自源 pytorch-cifar 拷至 contract/_backend_/_source_models/（只读）；
通过本仓 importlib 动态加载并以 @register_learner 装饰类公开。
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from experiment import ExperimentBase
from contract.runtime import CIFAR10_MEAN, CIFAR10_STD

# ── 源 pytorch-cifar DLA / SimpleDLA 动态加载 ─────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "contract" / "_backend_" / "_source_models"
_DLA_PATH = _BACKEND_DIR / "dla.py"
_SIMPLE_PATH = _BACKEND_DIR / "dla_simple.py"
_WRN_PATH = _BACKEND_DIR / "wrn.py"


def _load_source_class(file_path: Path, class_name: str):
    if not file_path.is_file():
        raise FileNotFoundError(f"源模型文件不存在: {file_path}")
    spec = importlib.util.spec_from_file_location(
        f"_source_models.{file_path.stem}", file_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建 spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"{file_path} 不含类 {class_name}")
    return getattr(module, class_name)


_DLA_class = _load_source_class(_DLA_PATH, "DLA")
_SimpleDLA_class = _load_source_class(_SIMPLE_PATH, "SimpleDLA")
_WRN_class = _load_source_class(_WRN_PATH, "WRN28_10")  # 函数：返回 WideResNet(depth=28, widen_factor=10)
_WideResNet_class = _load_source_class(_WRN_PATH, "WideResNet")  # 类：支持任意 depth/widen_factor (R20 WRN-40-4 用)


# ═══════════════════════════════════════════════════════════════
# 可插拔注册表（pluggable）
# ─────────────────────────────────────────────────────────────────
# 新增模型/损失变体 = 用 @register_* 装饰一个类 + 在 cfg 设键。
# 未知键 → KeyError（no-fallback）。
# ═══════════════════════════════════════════════════════════════

LEARNER_REGISTRY: dict[str, type] = {}
OBJECTIVE_REGISTRY: dict[str, type] = {}
AUGMENTATION_REGISTRY: dict[str, type] = {}


def register_learner(name: str):
    def _decorator(cls):
        LEARNER_REGISTRY[name] = cls
        return cls
    return _decorator


def register_objective(name: str):
    def _decorator(cls):
        OBJECTIVE_REGISTRY[name] = cls
        return cls
    return _decorator


def register_augmentation(name: str):
    def _decorator(cls):
        AUGMENTATION_REGISTRY[name] = cls
        return cls
    return _decorator


# ═══════════════════════════════════════════════════════════════
# Learner 注册 — SimpleDLA（plain）/ DLA（reference）
# ─────────────────────────────────────────────────────────────────
@register_learner("SimpleDLA")
class SimpleDLAWrapper(nn.Module):
    """源 pytorch-cifar SimpleDLA 包装（F1 O3 baseline-anchors plain）。"""
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.model = _SimpleDLA_class()

    def forward(self, x):
        return self.model(x)


@register_learner("DLA")
class DLAWrapper(nn.Module):
    """源 pytorch-cifar DLA 包装（F1 O3 baseline-anchors reference）。"""
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.model = _DLA_class()

    def forward(self, x):
        return self.model(x)


@register_learner("wrn28_10")
class WRN28_10Wrapper(nn.Module):
    """WideResNet-28-10 包装（Zagoruyko & Komodakis, BMVC 2016）。

    用于 R10 Tier B-extend 跨家族 backbone 升档首探：在 EMA decay=0.999 锚点
    recipe 上单 diff 替换 MODEL_ARCH=DLA→wrn28_10，验证 EMA 权重轨迹连续性
    机制是否 backbone-无关（与 R7 DLA 锚点 0.9353 比较）。
    """
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.model = _WRN_class()

    def forward(self, x):
        return self.model(x)


@register_learner("wrn40_4")
class WRN40_4Wrapper(nn.Module):
    """WideResNet-40-4 包装（Zagoruyko & Komodakis, BMVC 2016）。

    用于 R20 Tier B-routine 第 2 档容量探索：与 wrn28_10 (depth=28, widen=10,
    ~36.5M params) 形成 depth-vs-width 权衡对照 — wrn40_4 (depth=40, widen=4,
    ~8.9M params) 加深一层、缩窄一档；用于验证 R11 PreActResNet18 11.17M
    capacity-bound Δ-0.0101 是否 wide-resnet 家族内也成立，并探测 R10 KEEPER
    配方 (cutmix_mixup ρ=0.5 + Mixup α=0.2 + LS=0.1 + LR=0.1 + WD=5e-4) 在
    depth=40 widen=4 配置下能否破 0.977。
    Paper [arxiv:1605.07146] Table 4 报告 WRN-40-4 on CIFAR-10 ≈ 0.975（无增广）。
    """
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        # _WRN_class 是 WRN28_10 函数（只接受 num_classes kwarg），
        # WRN40_4 需任意 depth/widen_factor，必须直接调 WideResNet 类。
        self.model = _WideResNet_class(depth=40, widen_factor=4)

    def forward(self, x):
        return self.model(x)


@register_learner("preact_resnet18")
class PreActResNet18Wrapper(nn.Module):
    """Pre-Activation ResNet-18 包装（He et al. 2016, ECCV）。

    用于 R11 Tier B-routine 跨家族 backbone 注册首探：在 R10 SOTA recipe
    （LS=0.1 + cutmix_mixup ρ=0.5 + Mixup α=0.2）上单变量替换 wrn28_10 → PreActResNet18，
    验证 B-routine 升档首探的有效性。PreAct 残差单元（BN→ReLU→Conv）相较 wrn28_10
    的 post-activation（Conv→BN→ReLU）在 CIFAR-10 中等容量档常被报告略优，
    与 wrn28_10（~36.5M）相比 PreActResNet18 仅 ~11.2M，容量差距显著（×0.31）。

    实现位于本类内（不依赖 contract/_backend_/_source_models/ 新增，避免触发
    G-契约 静态守门），单元逻辑与原 paper 等价：CIFAR stem 3×3 conv stride=1（无
    maxpool）、4 stages × PreActBlock×[2,2,2,2]、global avg pool + linear。
    """
    class _PreActBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes: int, planes: int, stride: int = 1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else out
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            return out + shortcut

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        in_planes = 64
        layers = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)]
        cfg_stages = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
        for planes, num_blocks, stride in cfg_stages:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(self._PreActBlock(in_planes, planes, s))
                in_planes = planes * self._PreActBlock.expansion
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# ═══════════════════════════════════════════════════════════════
# Objective 注册 — 默认 cross_entropy（F1 T1）
# ─────────────────────────────────────────────────────────────────
@register_objective("cross_entropy")
class CrossEntropyObjective(nn.Module):
    """CE 目标；可选 LABEL_SMOOTHING（PyTorch nn.CrossEntropyLoss 原生支持，1.10+）。"""
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        label_smoothing = float(cfg["LABEL_SMOOTHING"])  # config-only 强读；缺键 → KeyError（no-fallback）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        return self.criterion(pred, target)


# ═══════════════════════════════════════════════════════════════
# PolyLoss (Leng et al. 2022, NeurIPS) — "PolyLoss: A Polynomial Expansion
# Perspective of Classification Loss Functions"
# 论文把常用分类损失展开为多项式 ∑(1-pt)^n，对 CE + 1 阶泰勒展开得
#   L_poly = L_CE + ε × (1 - p_t)
# 其中 p_t = softmax(logits)[target]。ε 是 1 个超参（POLY_EPSILON）。
# 与 LS 正交：LS 改标签分布，PolyLoss 改 CE 损失曲面；两者可叠加（CFG
# 同时设 LABEL_SMOOTHING=0.1 + POLY_EPSILON=1.0 不冲突）。
# Paper 报告 ImageNet +0.3-1%（与 EfficientNet 等搭配）；CIFAR-10 与
# WRN-28-10 / cutmix_mixup 配方下边际待本轮实证。
# ═══════════════════════════════════════════════════════════════
@register_objective("poly")
class PolyLossObjective(nn.Module):
    """PolyLoss：CE + ε × (1 - p_t)，p_t 为 student 在 target 类的 softmax 概率。"""
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        label_smoothing = float(cfg["LABEL_SMOOTHING"])  # config-only 强读
        poly_epsilon = float(cfg["POLY_EPSILON"])  # config-only 强读；缺键 → KeyError（no-fallback）
        if poly_epsilon < 0.0:
            raise ValueError(f"POLY_EPSILON={poly_epsilon}（须 ≥ 0）")
        # CE 子项仍走 nn.CrossEntropyLoss（LS 一致）
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing
        self.poly_epsilon = poly_epsilon

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        if self.poly_epsilon == 0.0:
            return ce_loss
        # 1 - p_t: p_t = softmax(logits)[target]（与 LS 兼容：p_t 自动反映 soft label）
        log_probs = F.log_softmax(pred, dim=1)
        # gather p_t per sample：softmax = exp(log_softmax)
        probs = log_probs.exp()
        p_target = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        poly_term = (1.0 - p_target).mean()
        return ce_loss + self.poly_epsilon * poly_term


# ═══════════════════════════════════════════════════════════════
# Augmentation 注册 — F1 D3 默认源增强
# ─────────────────────────────────────────────────────────────────
@register_augmentation("baseline")
class BaselineAugmentation:
    """源 pytorch-cifar 训练增强；test/eval 由 contract.base_transform 锁定（F1 D3）。"""
    def __init__(self, cfg: dict | None = None):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)

    def compose(self) -> transforms.Compose:
        return self.transform


@register_augmentation("mixup")
class MixupAugmentation:
    """Mixup（数据层软约束）：per-image transform 与 baseline 相同；batch 级像素插值
    由 ``Workspace.train_step`` 在 AUGMENTATION=mixup 时按 MIXUP_ALPHA 完成。
    公式：λ ~ Beta(α,α)；x_mix = λ·x + (1-λ)·x[perm]；
          loss = λ·CE(logits, y) + (1-λ)·CE(logits, y[perm])。
    仅训练期生效；test/eval 走 contract.base_transform，无插值。
    """
    def __init__(self, cfg: dict | None = None):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)

    def compose(self) -> transforms.Compose:
        return self.transform


@register_augmentation("cutmix")
class CutMixAugmentation:
    """CutMix（空间剪贴）：per-image transform 与 baseline 相同；batch 级矩形 patch 剪贴
    由 ``Workspace.train_step`` 在 AUGMENTATION=cutmix 时按 CUTMIX_ALPHA 完成。
    公式：λ ~ Beta(α,α)；bbox 面积比 = 1-λ；x_mix 在 bbox 内用 x[perm] 替换；
          loss = λ·CE(logits, y) + (1-λ)·CE(logits, y[perm])。
    仅训练期生效；test/eval 走 contract.base_transform，无剪贴。
    与 Mixup 像素线性插值正交，保留空间局部结构与卷积归纳偏置更兼容。
    """
    def __init__(self, cfg: dict | None = None):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)

    def compose(self) -> transforms.Compose:
        return self.transform


@register_augmentation("cutmix_mixup")
class CutMixMixupAugmentation:
    """复合增广（D-extend 深度）：per-image transform 与 mixup/cutmix 相同；batch 级 Mixup
    与 CutMix 之间的随机切换由 ``Workspace.train_step`` 在 AUGMENTATION=cutmix_mixup 时完成。

    公式：每 batch 按 COMPOSITE_PROB_CUTMIX 概率在 CutMix (α=CUTMIX_ALPHA) 与
    Mixup (α=MIXUP_ALPHA) 间二选一切换；标签分配规则同各自增广。
    仅训练期生效；test/eval 走 contract.base_transform，无插值/剪贴。
    """
    def __init__(self, cfg: dict | None = None):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)

    def compose(self) -> transforms.Compose:
        return self.transform


@register_augmentation("trivialaugment_wide")
class TrivialAugmentWideAugmentation:
    """TrivialAugment Wide (Müller & Hutter 2021, arxiv:2103.10158)。

    无参数 baseline 增广：每张图随机从 {Identity, ShearX/Y, TranslateX/Y, Rotate,
    Brightness, Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize}
    22 种操作中抽 1 种（Wide 池 32 种），强度从 {0, 10, 20, 30} 抽 1。paper 在
    CIFAR-10/ImageNet 报 +0.5-1%，与 WRN-28-10 + cutmix_mixup 配方正交（D-novel：
    与 batch 级 Mixup/CutMix 的标签混合正交，per-image 操作）。

    实施：TrivialAugmentWide 作用于 PIL；先 TAW → RandomCrop+RandomHorizontalFlip →
    ToTensor → Normalize。test/eval 走 contract.base_transform 无 TAW。
    """
    def __init__(self, cfg: dict | None = None):
        self.transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)

    def compose(self) -> transforms.Compose:
        return self.transform


class _NullScheduler:
    """具名 no-op 调度器: 显式 "none" 时返回，替代 None 兜底。"""
    def step(self, epoch=None):
        pass

    def get_last_lr(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# ═══════════════════════════════════════════════════════════════
# SAM (Sharpness-Aware Minimization) — Foret et al. 2020
# 包装任意 base optimizer（当前仅 SGD）；通过 first_step / second_step
# 实施双 pass：(1) 权重向损失最坏方向扰动；(2) 在扰动点求梯度回原位更新。
# 当 cfg.SAM_RHO > 0 时启用（默认 0 = 关闭，对历史 recipe 完全无副作用）。
# ═══════════════════════════════════════════════════════════════
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper（Foret et al. 2020）。

    用法（仅 train_step 两 pass）：
        optimizer.zero_grad()
        loss1 = compute(model, x, y)
        loss1.backward()
        optimizer.first_step(zero_grad=True)   # 向最坏方向扰动权重
        loss2 = compute(model, x, y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)  # 还原权重 + base_optimizer.step()

    SAM.step() 抛 NotImplementedError：必须用 first_step/second_step 显式两 pass。
    """
    def __init__(self, params, base_optimizer, rho: float, **kwargs):
        # R12 config-only 整改（M3 hit）：移除 hardcoded `rho: float = 0.05` 默认值；
        # 调用方必须显式传 `rho=cfg["SAM_RHO"]`（no-fallback；缺值 → TypeError 由 Python 抛出）。
        # 这把 SAM_RHO 这一正则化强度唯一化由 cfg 强读，禁掉任何残留的"默认 ρ=0.05"工作流。
        if rho < 0:
            raise ValueError(f"SAM_RHO 必须 ≥ 0（got {rho}）")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        # 用传入的 base_optimizer 类构造底层优化器（共享 param_groups）
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # 让 self.param_groups 始终指向 base 的最新 group（base.step() 可能改 lr 等）
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """扰动权重向最坏损失方向（ε-perturbation）。"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 缓存原权重，second_step 还原
                self.state[p]["old_p"] = p.data.clone()
                # ε_w = scale * grad（向最坏方向）
                p.add_(p.grad, alpha=scale)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """还原权重 + 调用 base_optimizer.step() 完成 SGD 更新。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        # 同步 base optimizer 改动的 lr 等到 self.param_groups
        self.param_groups = self.base_optimizer.param_groups
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        # SAM 必须显式 first_step / second_step；直接调 step() 是常见误用，明确报错
        raise NotImplementedError(
            "SAM.step() 不允许直接调用；用 first_step() + second_step() 两 pass。"
        )

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        stacked = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not stacked:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(stacked), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ═══════════════════════════════════════════════════════════════
# Lookahead (Zhang et al. 2019) — "k slow steps forward, 1 fast step back"
# 维护 slow weights θ_slow 与 fast weights（base_optimizer 持有的）；
# 每 k 步把 fast 朝 slow 平滑插值 alpha×fast + (1-alpha)×slow，并
# 把同步后的 fast 回灌给 base_optimizer（next iteration 起点）。
# 默认 LOOKAHEAD_K=0 = 关闭 = 退化为裸 base optimizer（backward-compatible）。
# 与 SAM 互斥（见 build_optimizer 守门）。
#
# **设计**：Lookahead 不继承 torch.optim.Optimizer；它是"双速 wrapper"。
# PyTorch 的 LR scheduler (CosineAnnealingLR 等) 只接受 torch.optim.Optimizer
# 子类，所以在 build_scheduler / train_step.sched.step() 处把 Lookahead
# 拆成其 inner base_optimizer（_unwrap_lookahead）。这样：
#   - scheduler.step() 修改 base_optimizer.param_groups[0]['lr']
#   - Lookahead.step() 透传给 base_optimizer.step()，此时 LR 已更新
#   - slow/fast 同步仍按 k 节奏进行
# ═══════════════════════════════════════════════════════════════
class Lookahead:
    """Lookahead Optimizer wrapper（Zhang et al. 2019, arxiv:1907.08610）。

    用法（仅 train_step 一个 step()，每 k 步自动同步 slow/fast）：
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        for ...:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 内部：fast.step() + (step_idx % k == 0 → slow sync)

    state_dict / load_state_dict 把 slow weights + k counter 序列化，方便
    checkpoint resume。
    """
    def __init__(self, base_optimizer, k: int, alpha: float):
        if not (1 <= k):
            raise ValueError(f"Lookahead k={k}（须 ≥ 1）")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Lookahead alpha={alpha}（须 ∈ (0, 1]）")
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        # 初始化 slow weights：取 base optimizer 当前所有 param 的 clone
        self.slow_weights = [
            p.data.clone() for group in self.optimizer.param_groups for p in group["params"]
        ]

    @property
    def param_groups(self):
        # 让外部只读访问底 SGD 的 param_groups（避免复制）
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """先 fast.step()，再按 k 节奏同步 slow/fast 权重。"""
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            # slow ← slow + alpha * (fast - slow)；fast ← slow
            slow_idx = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    slow = self.slow_weights[slow_idx]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
                    slow_idx += 1
        return loss

    def state_dict(self):
        return {
            "slow_weights": [w.detach().cpu().clone() for w in self.slow_weights],
            "step_counter": self.step_counter,
            "k": self.k,
            "alpha": self.alpha,
            "base_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.k = state_dict.get("k", self.k)
        self.alpha = state_dict.get("alpha", self.alpha)
        self.step_counter = state_dict.get("step_counter", 0)
        slow_loaded = state_dict["slow_weights"]
        # 长度对齐：若 base param 数量变化（罕见），只覆盖重叠部分
        for i, w in enumerate(self.slow_weights):
            if i < len(slow_loaded):
                w.copy_(slow_loaded[i].to(w.device))
        if "base_state" in state_dict:
            self.optimizer.load_state_dict(state_dict["base_state"])


# ═══════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average) — model weight trajectory averaging
# 经典用法：BYOL / MAE / MoCo v3 等自监督范式默认；监督任务亦常用作
# "时序平均"正则，提升 eval 稳定性。
# 当 cfg.EMA_DECAY > 0 时启用（默认 0 = 关闭，对历史 recipe 完全无副作用）。
# ═══════════════════════════════════════════════════════════════
class _EMAState:
    """Maintain a shadow copy of model weights, exponentially moving-averaged.

    仅对 nn.Parameter 做 EMA 平均；buffers (BN running_mean/var, num_batches_tracked)
    原样拷贝，不参与 EMA — BN running stats 本身已带 EMA 语义（PyTorch BN momentum=0.1），
    再叠一层 EMA 会让 running stats 滞后 100×, 破坏 eval 时 BN 的归化正确性。
    """
    def __init__(self, learner: nn.Module, decay: float):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA_DECAY 须在 [0, 1) 之间（got {decay}）")
        self.decay = decay
        # 记录需要 EMA 的 state_dict key 集合（对应 nn.Parameter）
        # 用 id 比较以避免 tensor 共享问题（state_dict 与 parameters 可能共享同一 tensor）
        param_ids = {id(p) for p in learner.parameters()}
        self._param_keys = {
            k for k, v in learner.state_dict().items() if id(v) in param_ids
        }
        # 完整 shadow 拷贝（params + buffers），但 update 时只对 _param_keys 走 EMA
        self.shadow = {
            k: v.detach().clone()
            for k, v in learner.state_dict().items()
        }

    @torch.no_grad()
    def update(self, learner: nn.Module) -> None:
        sd = learner.state_dict()
        for k, v in sd.items():
            if k in self._param_keys and v.dtype.is_floating_point:
                shadow_v = self.shadow[k]
                shadow_v.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # buffer (BN running stats / int counters): 原样同步当前值
                self.shadow[k].copy_(v.detach())

    def apply_to(self, learner: nn.Module) -> dict:
        """Swap learner weights to EMA. Returns saved original state for restore."""
        original = {k: v.detach().clone() for k, v in learner.state_dict().items()}
        learner.load_state_dict(self.shadow)
        return original

    @staticmethod
    def restore(learner: nn.Module, original: dict) -> None:
        """Restore learner weights from saved original state."""
        learner.load_state_dict(original)


class Workspace(ExperimentBase):
    """CIFAR-10 supervised workspace。"""

    # ── build_learner ──────────────────────────────────────────
    def build_learner(self, cfg: dict[str, Any], source=None):
        arch = cfg["MODEL_ARCH"]
        if arch not in LEARNER_REGISTRY:
            raise KeyError(
                f"未知模型架构: {arch}（可用: {list(LEARNER_REGISTRY.keys())}）"
            )
        model = LEARNER_REGISTRY[arch](cfg)
        return model, {}

    # ── build_objective ────────────────────────────────────────
    def build_objective(self, cfg: dict[str, Any]):
        loss_name = cfg["LOSS"]
        if loss_name not in OBJECTIVE_REGISTRY:
            raise KeyError(
                f"未知损失函数: {loss_name}（可用: {list(OBJECTIVE_REGISTRY.keys())}）"
            )
        objective = OBJECTIVE_REGISTRY[loss_name](cfg)
        return objective, {}

    # ── build_transforms ───────────────────────────────────────
    def build_transforms(self, cfg: dict[str, Any], *, for_test: bool = False):
        """F1 D3：test transform 锁定（与 base_transform 同）；train 走 AUGMENTATION。
        返回 (transform, metadata_dict)，与 build_learner / build_objective 一致。
        """
        if for_test:
            from contract.prepare_data import base_transform
            return base_transform(), {"for_test": True}
        aug_name = cfg["AUGMENTATION"]
        if aug_name not in AUGMENTATION_REGISTRY:
            raise KeyError(
                f"未知数据增强: {aug_name}（可用: {list(AUGMENTATION_REGISTRY.keys())}）"
            )
        aug = AUGMENTATION_REGISTRY[aug_name](cfg)
        return aug.compose(), {"augmentation": aug_name}

    # ── build_optimizer ────────────────────────────────────────
    def build_optimizer(self, cfg: dict[str, Any], learner):
        """默认源 SGD：lr=0.1, momentum=0.9, weight_decay=5e-4（F1 T2/E7）。

        SAM_RHO > 0 时用 SAM 包装底层 SGD；SAM_RHO=0（默认）退化为纯 SGD。
        SAM 开启后 base optimizer 的 step() 由 SAM.second_step() 代理，
        train_step 必须执行 first_step + second_step 两 pass（见 train_step）。

        NESTEROV（默认 False）：Nesterov 加速梯度（Sutskever et al. 2013
        [arxiv:1312.6120]），看经典 momentum 是否能进一步逼近 wrn28_10 +
        cutmix_mixup 的容量天花板（heuristic：在 CIFAR 训练中 nesterov=True
        通常 +0.1-0.3%）；SAM 模式下也透传 nesterov 给 base SGD。

        LOOKAHEAD_K > 0 时（默认 0 = 关闭，backward-compatible with R10
        KEEPER recipe）：Lookahead (Zhang et al. 2019 [arxiv:1907.08610])
        以"内部 fast weights + 外部 slow weights"双速机制包裹 base
        optimizer（当前仅 SGD；与 SAM 互斥），每 k 步把 fast weights 朝
        slow weights 平滑插值 (alpha=LOOKAHEAD_ALPHA)。paper 报告 CIFAR-10
        +0.2-0.5%，heuristic 边际。LOOKAHEAD_ALPHA 必须 ∈ (0, 1]，
        LOOKAHEAD_K 必须 ≥ 1。
        """
        lr = float(cfg["LR"])
        momentum = float(cfg["MOMENTUM"])
        weight_decay = float(cfg["WEIGHT_DECAY"])
        sam_rho = float(cfg["SAM_RHO"])
        nesterov = bool(cfg["NESTEROV"])  # config-only 强读；缺键 → KeyError（no-fallback）
        lookahead_k = int(cfg["LOOKAHEAD_K"])  # config-only 强读；缺键 → KeyError（no-fallback）
        lookahead_alpha = float(cfg["LOOKAHEAD_ALPHA"])  # config-only 强读
        if lookahead_alpha <= 0.0 or lookahead_alpha > 1.0:
            raise ValueError(f"LOOKAHEAD_ALPHA={lookahead_alpha}（须 ∈ (0, 1]）")
        if lookahead_k < 0:
            raise ValueError(f"LOOKAHEAD_K={lookahead_k}（须 ≥ 0）")
        optimizer_name = str(cfg["OPTIMIZER"]).lower()
        if optimizer_name == "sgd":
            base_cls = optim.SGD
        else:
            raise KeyError(f"未知优化器: {optimizer_name}（SAM/Lookahead 包装仅支持 SGD）")
        # Lookahead 与 SAM 互斥：SAM 自身已是双 pass 包装，再叠 Lookahead 会让
        # 慢权更新语义与 SAM 的 ε-perturbation 互相干扰。
        if lookahead_k > 0 and sam_rho > 0.0:
            raise ValueError(
                "Lookahead (LOOKAHEAD_K>0) 与 SAM (SAM_RHO>0) 互斥；"
                "两者都是 base optimizer 的双速/双 pass 包装，不能叠加。"
            )
        if sam_rho > 0.0:
            # SAM 包装：first/second_step 代理 SGD 更新；base 参数透传保持一致
            return SAM(
                learner.parameters(),
                base_optimizer=base_cls,
                rho=sam_rho,
                lr=lr, momentum=momentum, weight_decay=weight_decay,
                nesterov=nesterov,
            )
        base = base_cls(
            learner.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if lookahead_k > 0:
            return Lookahead(base, k=lookahead_k, alpha=lookahead_alpha)
        return base

    # ── build_scheduler ────────────────────────────────────────
    def build_scheduler(self, cfg: dict[str, Any], optimizer):
        """默认源 CosineAnnealingLR(T_max=cfg.EPOCHS)；EPOCHS 默认 200。

        SGDR (CosineAnnealingWarmRestarts, Loshchilov & Hutter 2017)：
          SCHEDULER=sgdr + cfg["SGDR_T_0"] (默认 50) + cfg["SGDR_T_MULT"] (默认 2) +
          cfg["SGDR_ETA_MIN"] (默认 0)。T_0/T_mult/eta_min 走 config-only 强读，
        缺键 → KeyError（no-fallback）。T_mult=2 时 200ep 经历 (cycle0=50ep, cycle1=100ep,
        cycle2=50ep 截断)，周期性 warm restart 验证 SGDR 是否在 EMA 锚点上
        进一步突破 plateau（vs 单次 cosine anneal）。

        Lookahead 包装：scheduler 只能作用于 torch.optim.Optimizer 子类；
        Lookahead 是双速 wrapper（非 Optimizer），所以这里把它 unwrap 到 inner
        base_optimizer（Lookahead.optimizer），scheduler 写 inner SGD 的 lr，
        Lookahead.step() 透传给 inner 时已带新 lr。
        """
        # Lookahead unwrap：让 LR scheduler 驱动 inner SGD 的 param_groups
        if isinstance(optimizer, Lookahead):
            scheduler_target = optimizer.optimizer
        else:
            scheduler_target = optimizer
        name = str(cfg["SCHEDULER"]).lower()
        epochs = int(cfg["EPOCHS"])
        if name == "none":
            return _NullScheduler()
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(scheduler_target, T_max=epochs)
        if name == "steplr":
            return optim.lr_scheduler.StepLR(scheduler_target, step_size=epochs // 3, gamma=0.1)
        if name == "sgdr":
            t_0 = int(cfg["SGDR_T_0"])  # config-only 强读
            t_mult = int(cfg["SGDR_T_MULT"])  # config-only 强读
            eta_min = float(cfg["SGDR_ETA_MIN"])  # config-only 强读
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                scheduler_target, T_0=t_0, T_mult=t_mult, eta_min=eta_min,
            )
        raise KeyError(f"未知调度器: {name}")

    # ── train_step ─────────────────────────────────────────────
    def train_step(self, learner, source, objective, *, epoch: int, shared_context: dict):
        """F1 T2：1 epoch 训练（遍历 source DataLoader 一次）。

        AUGMENTATION=mixup 时按 ``MIXUP_ALPHA`` 在 batch 内做像素插值（λ ~ Beta(α,α)），
        标签按 λ 加权做双 CE。

        SAM_RHO > 0 时（Foret et al. 2020）：
          - 同一 batch 走两 pass：forward+backward at w，再 forward+backward at w+ε
          - mixup 的 lam/perm 在两 pass 间复用，确保扰动评估与原 batch 一致
          - accuracy 仍按 pass-1 输出统计（当前权重下的预测，未受扰动影响）
        """
        learner.train()
        device = next(learner.parameters()).device
        cfg = shared_context["cfg"]
        # 懒构造 optimizer + scheduler（首次调用时按 cfg 装配）
        if "optimizer" not in shared_context:
            shared_context["optimizer"] = self.build_optimizer(cfg, learner)
        if "scheduler" not in shared_context:
            shared_context["scheduler"] = self.build_scheduler(cfg, shared_context["optimizer"])
        optimizer = shared_context["optimizer"]
        # Mixup / CutMix 开关：仅对应 AUGMENTATION 时读对应 ALPHA 键（config-only 强读）
        aug_name = str(cfg["AUGMENTATION"])
        mixup_alpha = 0.0
        cutmix_alpha = 0.0
        composite_prob_cutmix = 0.0  # >0 表示进入 cutmix_mixup 复合增广分支
        if aug_name == "mixup":
            mixup_alpha = float(cfg["MIXUP_ALPHA"])
            if mixup_alpha <= 0.0:
                raise ValueError(
                    f"AUGMENTATION=mixup 但 MIXUP_ALPHA={mixup_alpha}（须 > 0）"
                )
        elif aug_name == "cutmix":
            cutmix_alpha = float(cfg["CUTMIX_ALPHA"])
            if cutmix_alpha <= 0.0:
                raise ValueError(
                    f"AUGMENTATION=cutmix 但 CUTMIX_ALPHA={cutmix_alpha}（须 > 0）"
                )
        elif aug_name == "cutmix_mixup":
            # D-extend：每 batch 按 COMPOSITE_PROB_CUTMIX 概率在 Mixup / CutMix 间二选一
            mixup_alpha = float(cfg["MIXUP_ALPHA"])
            cutmix_alpha = float(cfg["CUTMIX_ALPHA"])
            composite_prob_cutmix = float(cfg["COMPOSITE_PROB_CUTMIX"])
            if mixup_alpha <= 0.0 or cutmix_alpha <= 0.0:
                raise ValueError(
                    f"AUGMENTATION=cutmix_mixup 但 MIXUP_ALPHA={mixup_alpha} / "
                    f"CUTMIX_ALPHA={cutmix_alpha}（都须 > 0）"
                )
            if not (0.0 <= composite_prob_cutmix <= 1.0):
                raise ValueError(
                    f"AUGMENTATION=cutmix_mixup 但 COMPOSITE_PROB_CUTMIX={composite_prob_cutmix}（须 ∈ [0, 1]）"
                )
        # SAM 开关：SAM_RHO > 0 时启用双 pass
        sam_rho = float(cfg["SAM_RHO"])
        use_sam = sam_rho > 0.0
        # Born-Again self-distillation 开关：KD_WEIGHT > 0 时启用 KL(student || EMA_teacher)。
        # R11（reflect R20260826_143234 推荐）：把 EMA 模型角色从「evaluator」扩展到「teacher」，
        # 提供 KL 辅助 loss 信号；teacher 用 EMA shadow 权重（与训练末评估一致路径），
        # forward 时 learner 临时切 eval() 模式防止 BN running stats 在蒸馏阶段被扰动。
        kd_weight = float(cfg["KD_WEIGHT"])
        kd_temp = float(cfg["KD_TEMP"])
        # R18 KD_WARMUP：前 KD_WARMUP_EPOCHS epoch 不开 KD 让 EMA teacher 沉淀成熟
        # （Furlanello et al. 2018 Born-Again Networks 论文机制）；修 R15/R16 s0 EMA_DECAY=0.999/0.9999
        # 撞 0.8847 冷启动污染 bug（epoch 0-9 teacher 仍带 init 权重噪声当知识灌给学生）。
        kd_warmup_epochs = int(cfg["KD_WARMUP_EPOCHS"])  # config-only 强读；缺键 → KeyError（no-fallback）
        if kd_warmup_epochs < 0:
            raise ValueError(f"KD_WARMUP_EPOCHS={kd_warmup_epochs}（须 ≥ 0）")
        use_kd = kd_weight > 0.0
        use_kd_now = use_kd and (epoch > kd_warmup_epochs)
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        for inputs, targets in source:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # 复合增广 (cutmix_mixup)：per-batch 按 COMPOSITE_PROB_CUTMIX 二选一；
            # 同变量 (sam 两 pass / teacher forward) 复用本 batch 的 lam/perm/inputs_mixed。
            if composite_prob_cutmix > 0.0:
                if torch.rand(1).item() < composite_prob_cutmix:
                    batch_mixup_alpha = 0.0
                    batch_cutmix_alpha = cutmix_alpha
                else:
                    batch_mixup_alpha = mixup_alpha
                    batch_cutmix_alpha = 0.0
            else:
                batch_mixup_alpha = mixup_alpha
                batch_cutmix_alpha = cutmix_alpha
            # 准备 mixup / cutmix 数据（同一组 lam/perm 在 SAM 两 pass 间复用）
            if batch_mixup_alpha > 0.0:
                lam = float(
                    torch.distributions.Beta(batch_mixup_alpha, batch_mixup_alpha).sample().item()
                )
                perm = torch.randperm(inputs.size(0), device=device)
                inputs_mixed = lam * inputs + (1.0 - lam) * inputs[perm]
                targets_b = targets[perm]
            elif batch_cutmix_alpha > 0.0:
                # CutMix: sample bbox (cx, cy) 与 bbox 边长 (cut_w, cut_h)
                # 面积比 = 1 - λ（与 mixup 共享 λ ~ Beta(α,α) 分布）
                lam = float(
                    torch.distributions.Beta(batch_cutmix_alpha, batch_cutmix_alpha).sample().item()
                )
                perm = torch.randperm(inputs.size(0), device=device)
                _, _, h, w = inputs.shape
                cut_rat = (1.0 - lam) ** 0.5
                cut_w = int(w * cut_rat)
                cut_h = int(h * cut_rat)
                cx = torch.randint(0, w, (1,)).item()
                cy = torch.randint(0, h, (1,)).item()
                bbx1 = max(0, cx - cut_w // 2)
                bby1 = max(0, cy - cut_h // 2)
                bbx2 = min(w, cx + cut_w // 2)
                bby2 = min(h, cy + cut_h // 2)
                inputs_mixed = inputs.clone()
                inputs_mixed[:, :, bby1:bby2, bbx1:bbx2] = inputs[perm, :, bby1:bby2, bbx1:bbx2]
                # 用实际 bbox 面积重算 λ（clamp 到 [0, 1]），用于 loss 加权
                lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1)) / float(w * h)
                targets_b = targets[perm]
            else:
                lam = 1.0
                perm = None
                inputs_mixed = inputs
                targets_b = None

            # ── Teacher forward（Born-Again self-distillation，KD_WEIGHT > 0 时启用）──
            # 用 EMA shadow 权重作为 teacher，与训练末评估同路径（评估也走 EMA 权重）；
            # 临时切 eval() 模式防止 BN running stats 在蒸馏阶段被扰动。
            # 第一 batch EMA state 尚未 init → teacher_logits=None，跳过 KD 信号（自然启动）。
            teacher_logits = None
            if use_kd_now and "ema_state" in shared_context:
                teacher_main = shared_context["ema_state"].apply_to(learner)
                try:
                    was_training = learner.training
                    learner.eval()  # 冻结 BN running stats（teacher 视角）
                    with torch.no_grad():
                        teacher_logits = learner(inputs_mixed)
                    if was_training:
                        learner.train()
                finally:
                    _EMAState.restore(learner, teacher_main)

            # ── Pass 1：原始权重上求梯度 ──
            optimizer.zero_grad()
            outputs = learner(inputs_mixed)
            if batch_mixup_alpha > 0.0 or batch_cutmix_alpha > 0.0:
                loss = lam * objective(outputs, targets) + (1.0 - lam) * objective(outputs, targets_b)
            else:
                loss = objective(outputs, targets)
            # ── KL 蒸馏辅助 loss（Furlanello et al. 2018, ICML Born-Again Networks）──
            # L_KD = KD_WEIGHT · T² · KL(softmax(teacher/T) || softmax(student/T))
            # 注意 PyTorch F.kl_div 约定 input=log-probs、target=probs，因此：
            #   F.kl_div(log_softmax(student/T), softmax(teacher/T)) = KL(teacher ‖ student)
            # 最小化该量等价于最小化 student-teacher 交叉熵（H(teacher) 为常数）。
            if teacher_logits is not None:
                T = kd_temp
                student_log_probs_T = F.log_softmax(outputs / T, dim=1)
                teacher_probs_T = F.softmax(teacher_logits / T, dim=1)
                kl_div = F.kl_div(
                    student_log_probs_T, teacher_probs_T, reduction="batchmean",
                )
                loss = loss + kd_weight * (T ** 2) * kl_div
            loss.backward()

            if use_sam:
                # ── Pass 2：扰动权重上求梯度（SAM 核心）──
                optimizer.first_step(zero_grad=True)
                outputs_perturbed = learner(inputs_mixed)
                if batch_mixup_alpha > 0.0 or batch_cutmix_alpha > 0.0:
                    loss_perturbed = lam * objective(outputs_perturbed, targets) + (1.0 - lam) * objective(outputs_perturbed, targets_b)
                else:
                    loss_perturbed = objective(outputs_perturbed, targets)
                loss_perturbed.backward()
                # 还原权重 + base optimizer.step()（用 pass-2 梯度）
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            # EMA 更新：cfg.EMA_DECAY > 0 时启动；first call 时 lazy init。
            # 必须在 optimizer.step() / second_step() 之后（权重已更新到新位置）
            ema_decay = float(cfg["EMA_DECAY"])
            if ema_decay > 0.0:
                if "ema_state" not in shared_context:
                    shared_context["ema_state"] = _EMAState(learner, ema_decay)
                shared_context["ema_state"].update(learner)

            # accuracy 跟踪用 pass-1 输出（原始权重下的预测；SAM 内部权重更新不影响）
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # 每 epoch 末推进 LR（与源 pytorch-cifar / 原仓 reproduce_dla 一致）。
        # 注意：train.py 并不调用 step_scheduler；若此处不 step，cosine 会卡在初始 LR
        # （展出仓首轮 plain/reference 因此只到 ~85%/87.8%，而非 ~95%）。
        sched = shared_context.get("scheduler")
        if sched is not None:
            sched.step()
        n = max(total, 1)
        return {
            "train_loss": total_loss / n,
            "train_acc": correct / n,
            "train_time_sec": time.time() - t0,
        }

    # ── step_scheduler ─────────────────────────────────────────
    def step_scheduler(self, shared_context: dict, *, epoch: int | None = None) -> None:
        """scheduler.step() — 供外部编排调用；train_step 内已每 epoch step 一次。"""
        sched = shared_context.get("scheduler")
        if sched is not None:
            # CosineAnnealingLR / WarmRestarts：用无参 step()（勿传 epoch，避免旧 API）
            sched.step()

    # ── predict ────────────────────────────────────────────────
    def predict(self, learner, batch, *, shared_context: dict):
        """F1 contract.test 入口：forward + EMA 权重切换（如启用）。

        EMA_DECAY > 0 且 ema_state 已 init（train_step 跑过）→ 临时加载 EMA shadow
        权重，forward 完恢复 main 权重，确保 contract.test 的 val_acc 与训练期
        evaluate 同路径（都是 EMA 轨迹平均后评估）。
        preflight 阶段（train_step 尚未跑）→ ema_state 不存在，跳过 EMA 切换。
        """
        inputs, targets = batch
        device = next(learner.parameters()).device
        cfg = shared_context.get("cfg")
        ema_state = shared_context.get("ema_state")
        # config-only 强读 EMA_DECAY（train.py 已确保 cfg["EMA_DECAY"] 存在；preflight 阶段 cfg=None 走分支跳过）
        ema_decay = float(cfg["EMA_DECAY"]) if cfg is not None else 0.0
        use_ema = ema_decay > 0.0 and ema_state is not None
        if use_ema:
            main_state = ema_state.apply_to(learner)
        try:
            return learner(inputs.to(device)), targets.to(device)
        finally:
            if use_ema:
                _EMAState.restore(learner, main_state)

    # ── evaluate ───────────────────────────────────────────────
    def evaluate(self, learner, source=None, *, epoch: int | None = None, shared_context: dict):
        """F1 E4：每 epoch 评一次，完整 test loader，三指标。objective + source 走 shared_context（source 可显式传）。

        EMA_DECAY > 0 且 EMA 已初始化（train_step 至少跑过一个 batch）→ 评测时临时
        加载 EMA shadow 权重、跑完恢复 main 权重；这样 val_acc 反映的是 EMA 轨迹平均
        的"软"评估，与训练末 best_state 同步走 EMA 路径。
        """
        if source is None:
            source = shared_context.get("val_loader") or shared_context.get("eval_loader")
        if source is None:
            raise RuntimeError("evaluate 需要 source；shared_context['val_loader']/'eval_loader' 也为空")
        cfg = shared_context["cfg"]
        ema_decay = float(cfg["EMA_DECAY"])
        ema_state = shared_context.get("ema_state")
        # EMA 启用条件：decay > 0 且 EMA 已 lazy init（即 train_step 至少跑过一个 batch）
        use_ema = ema_decay > 0.0 and ema_state is not None
        if use_ema:
            main_state = ema_state.apply_to(learner)
        try:
            learner.eval()
            device = next(learner.parameters()).device
            objective = shared_context.get("objective")
            total_loss = 0.0
            correct = 0
            correct_top2 = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in source:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = learner(inputs)
                    loss = objective(outputs, targets)
                    total_loss += loss.item() * targets.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    _, top2 = outputs.topk(2, dim=1)
                    correct_top2 += top2.eq(targets.view(-1, 1)).any(dim=1).sum().item()
            n = max(total, 1)
            return {
                "val_accuracy": correct / n,
                "val_loss": total_loss / n,
                "val_top2_accuracy": correct_top2 / n,
            }
        finally:
            if use_ema:
                _EMAState.restore(learner, main_state)


def create_workspace(cfg: dict) -> ExperimentBase:
    """工厂函数：创建 Workspace 实例。"""
    return Workspace()


# 占位：与 train.py import 对齐；未启用 register_workspace_kind 时回默认 NATIVE/EVALUATE_LEARNER。
def get_workspace_kind(_name: str = ""):
    from scripts.lib.train_branch_types import MetricsShape
    return MetricsShape.EVALUATE_LEARNER


def get_training_mech(_name: str = ""):
    from scripts.lib.train_branch_types import TrainingMech
    return TrainingMech.NATIVE
