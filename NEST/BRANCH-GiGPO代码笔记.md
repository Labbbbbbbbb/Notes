[GitHub - Labbbbbbbbb/NEST · GitHub](https://github.com/Labbbbbbbbb/NEST) main分支
框架基于GiGPO的verl-agent搭建，初步代码框架见GiGPO笔记

>[!NOTE]
>这是一份来自gpt的训练输出解读指南

按当前 `scripts/train_gigpo.sh alfworld` 启动，文件主要分成 **运行配置、指标日志、显存快照和训练 checkpoint**。其中 checkpoint 默认不在 `outputs/` 下。
下面用 `<运行名>` 表示：
```
gigpo_Qwen2.5-7B-Instruct_20261114_日期时间
```
实际使用“算法_模型_年月日_时分秒”的格式。
### 1. 运行配置与程序日志

```
outputs/<运行名>/
├── .hydra/
│   ├── config.yaml
│   ├── overrides.yaml
│   └── hydra.yaml
└── main_ppo.log
```

|文件|用途|
|---|---|
|`.hydra/config.yaml`|本次启动的训练配置快照，包含默认配置和覆盖后的参数；可能保留 `${...}` 插值表达式|
|`.hydra/overrides.yaml`|本次启动传入的 Hydra 覆盖项，例如模型、学习率、batch 等|
|`.hydra/hydra.yaml`|Hydra 自身的配置，例如输出目录、工作目录、日志设置|
|`main_ppo.log`|Python 日志系统写入的程序日志；**不等于完整控制台输出**，不保证包含 Ray worker 的所有打印和报错|
这些通常在训练入口启动时就产生。**修改里面的配置不会影响下一次训练**，它们是历史记录。
### 2. TensorBoard 指标与超参快照
```
outputs/tensorboard/<运行名>/
├── events.out.tfevents.…<run_id>
├── config.<run_id>.json
└── memory_snapshots/
    └── my_snapshot_….pickle
```

|文件|用途|
|---|---|
|`events.out.tfevents.…`|TensorBoard 指标数据，例如 loss、奖励、成功率、KL、耗时等；具体以训练实际记录的指标为准|
|`config.<run_id>.json`|训练 logger 初始化时的完整配置，插值已经展开；文件后缀与对应事件文件一致|
|`memory_snapshots/*.pickle`|新加的 CUDA 显存分配历史，供 Memory Viz 查看|
**JSON 配置与 Hydra YAML 的区别**：前者在训练循环开始时保存，能包含此前程序补充的配置；后者在启动时保存。因此模型初始化失败时，可能有 Hydra 配置，却没有 TensorBoard 事件和 JSON。
显存快照会更早启用，在 GPU worker 的相关方法执行前开始记录。每个阶段首次成功保存一次，异常时也尝试保存。文件名包含：

```
my_snapshot_<时间>_<主机>_rank<编号>_pid<进程号>_<阶段>.pickle
```
例如 `init_model_error` 表示初始化阶段异常。四卡会有多个 worker 各自的快照，**一个文件不代表四张卡的汇总**。日志会打印每份快照的绝对路径。
### 3. W&B 本地记录
```
outputs/wandb/<运行名>/
└── wandb/
    ├── run-…/ 或 offline-run-…/
    ├── latest-run
    └── …
```
具体子文件取决于 W&B 版本和在线/离线模式，常见内容包括：

|内容|用途|
|---|---|
|`run-*.wandb`|W&B 的本地运行记录，包含指标等数据，可用于同步|
|`logs/`|W&B 初始化、上传和同步的调试日志|
|`files/`|配置、运行元信息等文件；具体文件以实际生成情况为准|
|`latest-run`|指向最近一次 W&B 运行目录的链接|
在线模式还会上传到 W&B。当前初始化顺序是先 W&B、再 TensorBoard，所以 W&B 初始化失败时，TensorBoard 日志也可能尚未创建。
### 4. 模型 checkpoint：默认在 `checkpoints/`
```
checkpoints/nest/<运行名>/
├── latest_checkpointed_iteration.txt
└── global_step_25/
    ├── data.pt
    └── actor/
        ├── model_world_size_4_rank_0.pt
        ├── optim_world_size_4_rank_0.pt
        ├── extra_state_world_size_4_rank_0.pt
        ├── …其他 rank 的分片…
        └── …模型配置、tokenizer 等文件…
```

|文件|用途|
|---|---|
|`model_world_size_*_rank_*.pt`|各 GPU rank 保存的模型权重分片|
|`optim_world_size_*_rank_*.pt`|优化器状态，用于继续训练|
|`extra_state_world_size_*_rank_*.pt`|随机数等额外训练状态|
|`data.pt`|DataLoader 状态，用于恢复数据读取进度|
|`latest_checkpointed_iteration.txt`|最近保存的训练步数|
|模型配置、tokenizer 文件|模型结构和分词相关信息|

当前脚本设置 **每 25 个训练 step 保存一次**。GiGPO 当前不使用 critic，因此通常没有 `critic/`；启用 critic 的算法才会额外保存它。
默认保存的是训练分片，**不是一个可直接拿去推理的完整 Hugging Face 权重目录**。中途在初始化阶段 OOM，不会产生这些训练 checkpoint。
### 5. Slurm 控制台日志
默认位于项目根目录：
```
slurm-<作业号>.out
```
包含 shell 命令、程序打印、Ray worker 输出、警告和异常堆栈等。你现在查看的 `slurm-56693.out` 就是这一类。
它由 Slurm 保存，**不在默认 `outputs/` 目录中**，也不会自动改成算法和模型命名。
### 6. 你当前打开的 CSV 和图表
```
outputs/slurm-55183-plots/
├── metrics.csv
├── summary.json
├── overview.png
├── overview.svg
└── validation_tasks.png
```
这些是**此前对日志做分析后生成的产物（具体地说，是第一次完整跑完的GiGPO Qwen1.5B模型，wandb上传失败了，只能从本地文件自己画图分析），不是每次训练自动生成的**：
- `metrics.csv`：整理出来的指标表。
- `summary.json`：分析摘要。
- `overview.png/svg`：指标总览图。
- `validation_tasks.png`：验证任务相关图表。

`outputs/.write_test_*` 则属于写入测试文件，不是训练指标或模型产物。

另外，当前 `rollout_data_dir`、`validation_data_dir` 都是 `null`，所以**默认不会逐条保存训练和验证的完整输入、模型回答及奖励 JSONL**。有指标日志，并不代表每条交互轨迹都已落盘。