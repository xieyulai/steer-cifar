"""数据契约入口（CIFAR-10：train / test loader，train_eval 复用 test split）。"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from contract.runtime import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR10_TAR_PATH,
    DEFAULT_VAL_LOADER_BATCH_SIZE,
    LOCKED_DATASET,
)

if TYPE_CHECKING:
    from experiment import ExperimentBase

# ── Test split transform（与官方一致；不可改，F1 D3 锁定）────────
_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def base_transform() -> transforms.Compose:
    """与 test 集相同的归一化（train 可在 cfg 覆盖 _train_transform）。"""
    return _EVAL_TRANSFORM


# ── Train split transform（默认源增强；可经 cfg 覆盖，F1 D3 锁定 baseline）──
_TRAIN_TRANSFORM_DEFAULT = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def default_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ── 数据集注册表(范式锁死 / contract-frozen)──────────────────────
# 要求2:读哪个数据集 = LOCKED_DATASET(contract 常量,Agent 不可达);
#   怎么处理(transform/aug/batch)可改 = 走 cfg(见下方 DataLoader 参数)。
# 新增数据集 = 在 DATASET_REGISTRY 加一行(contract 改,非 agent 每轮动作)。
# 未知名 → KeyError(no-fallback;不静默回落)。
DATASET_REGISTRY: dict = {
    "cifar10": lambda root, train, transform: datasets.CIFAR10(
        root, train=train, download=False, transform=transform,
    ),
    # 兼容老调用；如果有人改 LOCKED_DATASET 回 fashionmnist 也可工作
    "fashionmnist": lambda root, train, transform: datasets.FashionMNIST(
        root, train=train, download=True, transform=transform,
    ),
}


def _build_dataset(name: str, root: str, *, train: bool, transform):
    key = str(name).lower().strip()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"DATASET={name!r} 不支持;"
            f"新增 = 在 DATASET_REGISTRY 加一行(contract 改,非 agent 每轮动作);"
            f"可用: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[key](root, train, transform)


def _ensure_cifar10_extracted(data_dir: str) -> None:
    """F1 D4: 若 data_dir 缺 cifar-10-batches-py，则从 CIFAR10_TAR_PATH 解压。"""
    data_root = Path(data_dir)
    expected = data_root / "cifar-10-batches-py"
    if expected.is_dir():
        return
    tar_path = Path(CIFAR10_TAR_PATH)
    if not tar_path.is_file():
        raise FileNotFoundError(
            f"CIFAR-10 未解压且 tar 包不可读: 期望 {expected} 或源 {tar_path}"
        )
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_root)
    if not expected.is_dir():
        raise RuntimeError(
            f"CIFAR-10 解压后未发现 {expected}；请检查 tar 内容"
        )


def prepare_data(contract: ExperimentBase, cfg: dict) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10：按 F1 D2 严格 source-exact：
        - train_loader: 50000 张 train split + 默认源增强 (RandomCrop+RandomHorizontalFlip+normalize)
        - eval_loader:  10000 张 test split  + eval transform（每 epoch 评；F1 E4）
    不再保留内部 val 切分（F1 D2：VAL=none）。
    """
    batch_size = int(cfg["BATCH_SIZE"])
    num_workers = int(cfg.get("NUM_WORKERS", 2))
    pin = torch.cuda.is_available()

    data_dir = contract.data_dir
    # F1 D4: 缺数据时解压本机 tar 包
    _ensure_cifar10_extracted(data_dir)

    train_dataset = _build_dataset(
        LOCKED_DATASET, data_dir, train=True, transform=default_train_transform(),
    )
    test_dataset = _build_dataset(
        LOCKED_DATASET, data_dir, train=False, transform=base_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_VAL_LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, eval_loader


def prepare_shared_context(contract: ExperimentBase, shared_context: dict) -> None:
    """任何 profile：将数据根写入 shared_context（供 contract.test / runner）。"""
    raw = os.environ.get("NN_DATA_DIR", "").strip()
    if raw:
        shared_context["dataset_path"] = str(Path(raw).expanduser().resolve())
    else:
        data_dir = getattr(contract, "DATA_DIR", None) or contract.data_dir
        shared_context["dataset_path"] = str(Path(data_dir).expanduser().resolve())
