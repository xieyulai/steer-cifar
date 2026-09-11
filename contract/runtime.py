"""契约常量：数据根、归一化、eval 规模等（供 prepare_data / test / 迁移参照）。"""
from __future__ import annotations

import os
from pathlib import Path

# ── 数据目录配置 ─────────────────────────────────────────────────
# 优先级：NN_DATA_DIR 环境变量 > DEFAULT_DATA_DIR > data/
# 数据量小（<1GB）：拷贝到 data/ 目录
# 数据量大（>=1GB）：软链接指向原始数据路径
#   ln -s /path/to/original/data data/dataset_name
DEFAULT_DATA_DIR = os.environ.get("NN_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

# ── Supervised (CIFAR-10 迁入) ──────────────────────────────────
# LOCKED_DATASET = 范式锁死的数据集身份(要求2):由 contract 钉死,不走 cfg,
# Agent 每轮动作不可达(改哪个数据集 = E 档/人审级,非 A-D 轮内动作)。
# 怎么处理(transform/aug/batch)仍可改 = 走 cfg(见 prepare_data 下游 DataLoader)。
# 新增数据集 = 在 prepare_data 的 DATASET_REGISTRY 加一行 + 改本常量。
LOCKED_DATASET = "cifar10"
# CIFAR-10 官方 channel-wise mean/std（与源 pytorch-cifar 完全一致）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
# 训内/官方 test eval 都用同一批大小；源 main.py testloader batch=100
DEFAULT_VAL_LOADER_BATCH_SIZE = 100

# ── F1 D4: 复现数据锁本机 tar 包（缺数据时由 prepare_shared_context 解压）──
CIFAR10_TAR_PATH = os.environ.get(
    "NN_CIFAR10_TAR",
    "/data/xieyl/xie_workspace/AUTO_RESEARCH/CIFAR/cifar-10-python.tar.gz",
)

# ── 复现环境变量（HARD-GATE D4/O4 钉死；finalize 写入 config.json `_repro`）──
REPRO_ENV_KEYS: tuple[str, ...] = (
    "NN_SEED",
    "NN_DATA_DIR",
    "CUDA_VISIBLE_DEVICES",
)

# ── 信息权限登记表（立项三问；必须字面量）────────────────────────
INFO_PERM = {
    "enforce": True,
    "official_path": 'full_model',        # full_model | restricted | eval_mode_locked（问②）
    "official_path_impl": None,           # restricted / eval_mode_locked 时 run() 必须调用的函数名
    "official_path_switch": "_official_path_enabled",   # 交卷开关函数名；存在则须单条 return True/False
    "eval_only_assets": (),               # 问①：只评资产——路径 glob，或 contract. 开头的读取函数名
    "eval_only_readers": ('contract/test.py',),          # 允许打开只评资产的文件（仓根相对路径）
    "train_batch_keys": None,             # 训练首 batch 键白名单（dict 键 / tuple 项数）；None 不校
    "strict_no_train_files": False,       # D2 TRAIN 为「无数据文件」时置 True：工作区不得读任何数据文件
}
