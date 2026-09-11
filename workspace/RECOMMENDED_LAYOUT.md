# workspace 起步布局建议（profile=supervised）

> **仅命名建议，不参与任何 gate。** 文件名完全自由：`model.py` 改叫 `models/`、
> 把 `loss.py` 并进 `train_loop.py` 都不影响验收。守门只校验 `Workspace` 类上的
> **方法**（见下）是否存在，不看文件叫什么。可删除本文件。

## 必须由 `Workspace` 暴露的方法（G-范式 / G-门面 校验）

- `build_learner`
- `build_objective`
- `train_step`
- `predict`
- `evaluate`

## 推荐子模块命名（可改可不用）

- `workspace/model.py` — model
- `workspace/data_process.py` — data
- `workspace/loss.py` — loss
- `workspace/train_loop.py` — train
- `workspace/infer.py` — infer

## 不变量（迁移铁律）

- `train.py` 只调 `ws.*` / `contract.*`，**禁止** `from workspace.<子模块> import`（G-封装）。
- 数据划分 / 官方 test / metrics 归 `contract/`，不进 workspace。

