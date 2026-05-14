---
name: implement-logging
description: Implements Python logging (handlers, levels, file + console, training metrics) and replaces ad-hoc print. Use when the user asks for 日志、日志系统、logging、log file、记录训练、或把 print 改成日志。
disable-model-invocation: true
---

# 实现日志记录系统

## 何时启用

用户要**接入标准 logging**、统一训练/脚本的输出、落盘可追溯、或排查分布式/多进程输出混乱时，按本节执行。

## 原则

- **入口配置一次**：在 `main()` 或 CLI 入口调用配置函数，避免在 import 时副作用写文件。
- **库代码用 `logging.getLogger(__name__)`**，不设 `basicConfig`；**可执行脚本**在入口设 root 或具名 logger 的 handler。
- **级别**：默认 `INFO`；调试细节用 `DEBUG`；可恢复问题用 `WARNING`，真正失败用 `ERROR` + `exc_info=True`。
- **敏感信息**：不向日志写入密钥、token、完整路径中的用户名（若敏感）、或未经脱敏的 PII。

## 推荐布局（本仓库）

- 日志目录：`src/logs/`（与现有 `log.txt` 同层）；大文件或频繁写入时用 `RotatingFileHandler` 或按运行子目录 `src/logs/runs/<timestamp>/`.
- 模块内：`logger = logging.getLogger(__name__)`，在训练循环、数据加载、保存 checkpoint 等处打 `logger.info`。
- 与 `patchtst_impute/train.py` 一致：用 `Path(__file__).resolve().parents[...]` 定位仓库根，再拼 `src/logs/...`，路径字符串在代码里用**正斜杠**或 `Path` 拼接。

## 最小实现步骤

1. **新增** `patchtst_impute/logging_utils.py`（或等价小模块）：`setup_logging(log_dir: Path, level: str, log_name: str) -> None`，完成：
   - `logging.Formatter`：`%(asctime)s %(levelname)s [%(name)s] %(message)s`，`datefmt` 含秒。
   - `StreamHandler(sys.stdout)` → 终端。
   - `logging.handlers.RotatingFileHandler`（或 `FileHandler`）→ `log_dir / log_name`，`encoding="utf-8"`。
   - `force=True`（Python 3.8+ 无则先 `clear` root handlers）避免重复 handler。
2. **CLI 增加** `--log_level`、`--log_dir`（默认 `src/logs`）、可选 `--run_name`。
3. **`main()` 开头**调用 `setup_logging`；将原 `print(...)` 改为 `logger.info(...)`；异常路径 `logger.exception(...)`。
4. **`.gitignore`**：若日志很大或含环境信息，忽略 `src/logs/*.log` 或整个 `src/logs/`（按团队约定）；不要把唯一调试证据误忽略。

## 训练脚本约定

- 每个 epoch 一行摘要：`epoch、train_mse、val_mse、lr、耗时` 等标量，便于 `grep`/对比。
- **避免**每 step 默认 `INFO` 刷屏；高频用 `logger.debug` 或每 N step 聚合一条 `INFO`。
- `torch`/`CUDA` 版本与关键超参在**训练开始**打一条 `INFO`。

## 多进程与 DataLoader

- `num_workers > 0` 时，子进程默认继承配置可能重复写文件或乱序；优先**仅在主进程**打结构化训练日志；worker 内尽量不打文件日志，必要时用 `warnings` 或主进程汇总。

## 反模式

- 在多个模块重复 `basicConfig` 导致 handler 叠加、同一条日志多次出现。
- 用 `print` 与 `logging` 混用且无统一格式。
- 超大 tensor / 整表 `logger.info(str(tensor))` 撑爆日志与 I/O。

## 可选扩展

- 需要 JSON/可解析流水线：对机器消费行使用 `json.dumps` 单行 `INFO`，人读仍保留文本摘要。
- 与实验跟踪（W&B、MLflow）并存时：**文件日志保留本地可审计副本**，tracker 只传标量与小 artifact。
