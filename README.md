# icl-impute

面向**多变量序列数据补全**的研究型代码库：在完整观测的 CSV 时间序列上，以随机点遮蔽做自监督训练，使用 **PatchTST 风格**的编码器学习补全被遮蔽位置的值。适用于探索「上下文式」补全表征与训练流程；默认管线为滑动窗口 + 按时间切分的训练 / 验证 / 测试。

## 环境要求

- **Python**：建议 3.11（与 `environment.yml` 一致）；3.10+ 一般可用。
- **硬件**：CPU 可跑；有 NVIDIA GPU 时可安装 CUDA 版 PyTorch 以加速（见 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)）。
- **依赖**：见仓库根目录 `requirements.txt`（`torch`、`numpy`、`tqdm`）。

## 数据格式

- 多变量 CSV，数值列用逗号分隔；首列若表头为 **`date`**（不区分大小写），则该列仅作时间索引、**不参与建模**。
- 训练入口会校验 `--data_path` 指向的文件是否存在；仓库内**不包含**示例数据，请自备 CSV 或使用例如 ETT 等公开数据集，放到例如 `test_data/ETT-small/ETTh1.csv` 后把路径传给 `--data_path`。

## 安装

以下均在**仓库根目录**执行。任选一种环境管理方式即可。

### 使用 uv

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

（Linux / macOS：`source .venv/bin/activate`）

### 使用 conda

```powershell
conda env create -f environment.yml
conda activate icl-impute
```

该环境会通过 pip 安装 `requirements.txt` 中的依赖。

### 使用 pip（venv）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

若需 GPU，请在激活虚拟环境后，按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择与你的 CUDA 版本匹配的 `pip install torch ...` 命令（可能覆盖纯 CPU 的 torch 轮子）。

## 启动训练

在仓库根目录激活上述任一环境后：

```bash
python src/impute/train.py --data_path 你的数据.csv
```

或使用模块方式（需保证能解析包 `impute`，例如已将 `src` 加入 `PYTHONPATH`）：

**PowerShell**

```powershell
$env:PYTHONPATH = "src"
python -m impute.train --data_path 你的数据.csv
```

**bash**

```bash
export PYTHONPATH=src
python -m impute.train --data_path 你的数据.csv
```

常用参数（默认值见 `--help`）：`--seq_len`、`--epochs`、`--batch_size`、`--device`（`cuda` / `cpu`）、`--data_path`。查看全部选项：

```bash
python src/impute/train.py --help
```

日志默认写入 `src/logs/train.log`（可用 `--log_dir`、`--log_file` 修改）。

## 项目结构

```bash
src/impute/
├── __init__.py             # 对外聚合导出
├── train.py                # 命令行入口CLI：argparse → TrainConfig → run_training
├── io/                 # 数据读入
│   ├── __init__.py  
│   └── csv.py              # load_multivariate_csv
├── data/               # 数据处理 - 时间切分、滑动窗口
│   ├── __init__.py
│   ├── splits.py           # SplitConfig, time_splits
│   └── windows.py          # SlidingWindowDataset, iter_batches
├── models/             # PatchTST 补全网络
│   ├── __init__.py
│   └── patchtst.py         # PatchTSTImputer 及辅助算子
├── training/           # 训练配置、DataLoader 组装、损失与遮蔽、训练引擎
│   ├── __init__.py
│   ├── config.py           # TrainConfig（超参数据类）
│   ├── dataloaders.py      # resolve_data_path, build_time_series_loaders
│   ├── masking.py          # make_train_mask
│   ├── losses.py           # masked_mse
│   └── engine.py           # setup_logging + evaluate + run_training
└── log_utils/          # 日志初始化（控制台 + 追加写文件）
    └── __init__.py         # setup_logging（包名避免与标准库 logging 冲突）

src/impute/train.py        # 命令行入口
requirements.txt、environment.yml   # 依赖声明
```

## 许可证与引用

若本仓库后续添加 `LICENSE` 或论文引用信息，请在此补充对应文件名与 BibTeX。
