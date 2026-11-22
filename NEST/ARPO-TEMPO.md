下面解释的是当前 NEST 中适配后的工程结构。

**一、两个工作区的总体结构**

```
NEST/
├── GiGPO/                         # main 分支、共享 Git 仓库
├── TAPO/                          # TAPO 分支
├── ARPO/                          # ARPO 分支
│   ├── arpo/                      # ARPO 专属算法实现
│   ├── scripts/
│   ├── examples/
│   ├── agent_system/
│   ├── verl/
│   ├── gigpo/
│   ├── tests/
│   └── docs/
└── TEMPO/                         # TEMPO 分支
    ├── tempo/                     # TEMPO 专属算法实现
    ├── scripts/
    ├── examples/
    ├── agent_system/
    ├── verl/
    ├── gigpo/
    ├── tests/
    └── docs/
```

两者沿用相同的训练框架，主要差别在于：

|工程部分|ARPO|TEMPO|
|---|---|---|
|轨迹采样|根据熵变化，自适应增加分叉|普通多轨迹采样|
|环境操作|分叉时需要恢复相同环境状态|正常执行环境动作|
|优势计算|每条完整轨迹计算组内相对优势|根据共享 token 前缀计算不同位置的优势|
|模型更新|共用 verl 的策略更新与 KL 约束|共用 verl 的策略更新与 KL 约束|

**二、`scripts/`：提交训练的入口**

重点文件是：

- [ARPO/scripts/train_arpo.sh](/data/home/zdhs0200/NEST/ARPO/scripts/train_arpo.sh)
- [TEMPO/scripts/train_tempo.sh](/data/home/zdhs0200/NEST/TEMPO/scripts/train_tempo.sh)

它们负责申请 Slurm 资源、激活 NEST 环境、进入正确工作区、设置 Python 路径、生成实验名称和日志目录，最后调用具体训练脚本。

默认申请 **4 张 GPU、32 个 CPU(已经删掉)、256G 内存**。显式设置的 `PYTHONPATH` 也会传给 Ray，保证 worker 使用对应工作区中的实现。

运行方式：

```
cd /data/home/zdhs0200/NEST/ARPO
sbatch scripts/train_arpo.sh
```

```
cd /data/home/zdhs0200/NEST/TEMPO
sbatch scripts/train_tempo.sh
```

目录中的模型合并、诊断等脚本来自原框架，不是算法的核心实现。

**三、`examples/`：具体实验的参数配置**

各自的主要文件是：

```
examples/
├── arpo_trainer/run_alfworld.sh    # ARPO 工作区
# 或
├── tempo_trainer/run_alfworld.sh   # TEMPO 工作区
└── data_preprocess/prepare.py
```

`run_alfworld.sh` 集中设置模型、学习率、batch、显存预算、环境步数、采样数量和评估频率。你以后调整实验参数，主要看这里。

两个分支目前的共同默认值包括：

| 参数                      | 默认值                 |
| ----------------------- | ------------------- |
| 模型                      | Qwen2.5-7B-Instruct |
| 训练任务数 / 验证任务数           | 16 / 128            |
| 每任务轨迹预算                 | 8                   |
| 最大 prompt / response 长度 | 2048 / 512          |
| 单条轨迹最大环境步数              | 50                  |
| 学习率                     | `1e-6`              |
| micro batch / GPU       | 8                   |
| vLLM 显存利用率              | 0.5                 |

`data_preprocess/prepare.py` 准备训练框架需要的数据记录。**实际的 ALFWorld 任务和观测来自环境交互**，不是用其加载的几何题内容训练。

数据输出分别放在各工作区的 `outputs/data/text/`，避免两个工作区相互覆盖。

**四、`arpo/`：ARPO 的专属实现**

```
arpo/
├── __init__.py
├── core_arpo.py
└── rollout.py
```

[core_arpo.py](/data/home/zdhs0200/NEST/ARPO/arpo/core_arpo.py) 负责三件事：计算上游使用的熵统计量、根据熵变化计算分叉概率，以及根据完整轨迹结果计算组内优势。

这里按“轨迹”统计奖励，一条轨迹即使有很多回合，也只参与一次组内均值和标准差计算，避免长轨迹被重复计数。

[rollout.py](/data/home/zdhs0200/NEST/ARPO/arpo/rollout.py) 负责实际执行自适应采样：

1. 每个任务先激活 4 条轨迹。
2. 根据生成结果的熵变化决定是否分叉。
3. 分叉时复制环境状态和 agent 历史。
4. 最终收集每组预算内的 8 条轨迹。

目前默认 `beam_size=2`、基础分叉概率 `0.5`、熵变化权重 `0.2`。如果所有活动轨迹都结束而预算尚未用完，会从原任务起点继续采样。

**五、`tempo/`：TEMPO 的专属实现**

```
tempo/
├── __init__.py
└── core_tempo.py
```

[core_tempo.py](/data/home/zdhs0200/NEST/TEMPO/tempo/core_tempo.py) 负责从采样结果建立共享前缀树，并计算 token 级优势。

例如，同一任务的三条响应为：

```
共同前缀
├── 动作 A → 最终成功
├── 动作 A → 最终失败
└── 动作 B → 最终失败
```

TEMPO 会利用共享前缀下各条轨迹的最终结果，估计前缀价值；不同 token 分支因而可以得到不同的优势。

NEST 把每个 assistant 回合保存成一行，因此这里额外处理了跨回合的前缀关系，并加入回合边界，避免把不同动作序列误认为相同前缀。

这个适配使用普通轨迹采样，前缀树在优势计算阶段建立。

**六、`agent_system/`：agent 与环境如何交互**

```
agent_system/
├── environments/
├── memory/
├── multi_turn_rollout/
└── reward_manager/
```

|子目录|主要职责|
|---|---|
|`environments/`|创建 ALFWorld 等环境，解析模型动作，生成观测和提示词|
|`memory/`|保存每个环境的动作、观测历史，构造下一回合输入|
|`multi_turn_rollout/`|循环执行“生成动作 → 环境执行 → 获取反馈”，整理训练数据|
|`reward_manager/`|将 episode 奖励整理为训练框架使用的奖励|

ARPO 在这里有一个特别重要的文件：

[environments/env_package/alfworld/envs.py](/data/home/zdhs0200/NEST/ARPO/agent_system/environments/env_package/alfworld/envs.py)

新增的 `snapshot`、`restore`、`fork` 用来记录当前任务和已执行动作，并在目标 worker 上重放同一动作前缀。恢复后会检查观测是否一致，同时保留目标 worker 后续正常选择任务的顺序。

这部分已通过本地真实 ALFWorld 测试。目前支持**文本 ALFWorld，且关闭 domain randomization**。

**七、`verl/`：训练调度与 GPU 执行框架**

```
verl/
├── trainer/
├── workers/
├── models/
├── single_controller/
├── utils/
└── protocol.py
```

|部分|功能|
|---|---|
|`trainer/`|组织训练、计算优势、评估、保存检查点|
|`workers/`|执行模型生成、概率计算和梯度更新|
|`models/`|模型结构与后端适配|
|`single_controller/`|Ray worker、资源池和远程调用管理|
|`utils/`|数据集、日志、张量处理、检查点等工具|
|`protocol.py`|定义 `DataProto`，携带张量、轨迹 ID 和其他元数据|

最值得阅读的几个文件是：

- `trainer/main_ppo.py`：读取配置，创建训练组件。
- `trainer/ppo/ray_trainer.py`：组织一轮训练，并接入 ARPO/TEMPO 优势。
- `trainer/config/ppo_trainer.yaml`：框架默认参数。
- `workers/actor/dp_actor.py`：计算策略损失并反向传播。
- `workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：执行 vLLM 生成；ARPO 在此获取分叉所需的熵统计。

文件名中的 `ppo` 表示沿用的训练框架，实际算法由 `algorithm.adv_estimator` 决定。

**八、`gigpo/`、`tests/`、`docs/` 与运行产物**

`gigpo/` 保留原有 GiGPO 算法实现，供原算法路径使用。ARPO、TEMPO 的启动入口分别选择自己的优势估计器。

`tests/arpo/` 当前有 9 项测试，覆盖公式、分叉、环境重放和训练数据管线；`tests/tempo/` 有 5 项测试，覆盖前缀优势、回合边界、补齐和重排。

运行后会生成：

```
outputs/
├── data/text/             # 数据占位文件
├── <实验名>/              # Hydra 运行配置等
├── wandb/<实验名>/        # W&B 本地日志
└── tensorboard/<实验名>/  # TensorBoard 日志

checkpoints/
└── nest/<实验名>/         # 模型训练检查点
```

详细说明已保存到：

- [ARPO 工程与运行说明](/data/home/zdhs0200/NEST/ARPO/docs/arpo_nest.md)
- [TEMPO 工程与运行说明](/data/home/zdhs0200/NEST/TEMPO/docs/tempo_nest.md)

阅读代码可以按这个调用顺序：

```
scripts/train_*.sh
  → examples/*_trainer/run_alfworld.sh
  → verl/trainer/main_ppo.py
  → ray_trainer.py
  → agent_system 收集轨迹
  → arpo/ 或 tempo/ 计算优势
  → workers/actor/dp_actor.py 更新模型
```