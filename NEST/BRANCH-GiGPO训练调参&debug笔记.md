[GitHub - Labbbbbbbbb/NEST · GitHub](https://github.com/Labbbbbbbbb/NEST) main分支
框架基于GiGPO的verl-agent搭建，初步代码框架见GiGPO笔记

基础配置yaml文件：NEST/verl/trainer/config/ppo_trainer.yaml
命令行脚本：NEST/examples/gigpo_trainer/run_alfworld.sh
参数解析时拿到的config文件，里面的参数是在yaml的基础上被命令行覆盖过的

# 主要参数含义

| 配置                                   | 当前值   | 含义                                         |
| ------------------------------------ | ----- | ------------------------------------------ |
| `data.train_batch_size`              | 16    | 每轮 rollout 取 4 个任务                         |
| `env.rollout.n`                      | 128   | 每个任务采样 8 条完整轨迹                             |
| `env.max_steps`                      | 50    | 每条轨迹最多执行 50 次环境交互                          |
| `actor.ppo_epochs`                   | 1（默认） | 每批 rollout 数据用于训练 1 遍                      |
| `actor.ppo_mini_batch_size`          | 256   | 每个训练 mini-batch 的全局样本数                     |
| `actor.ppo_micro_batch_size_per_gpu` | 32    | 每张 GPU 每次前向/反向处理 2 条样本，通过梯度累积组成 mini-batch |
| `log_prob_micro_batch_size_per_gpu`  | 32    | 重新计算旧策略 log probability 的分块大小              |
其中两个 `micro_batch_size` 主要影响显存和计算效率，`train_data_size` 会直接改变每轮采样的数据量，从而影响训练。

| 参数                                  | 在当前项目中的作用                                               | 对训练效果的影响                         |
| ----------------------------------- | ------------------------------------------------------- | -------------------------------- |
| `log_prob_micro_batch_size_per_gpu` | 每张 GPU 一次处理多少条样本来计算 log probability                     | 通常不直接改变学习目标，主要影响显存和速度            |
| `ppo_micro_batch_size_per_gpu`      | 每张 GPU 一次前向、反向处理多少条训练样本                                 | 固定 mini-batch 时，通常通过梯度累积保持更新近似一致 |
| `train_data_size`                   | 同时设置训练数据占位条数和 `data.train_batch_size`，决定每轮 rollout 的任务数 | 会改变采样规模、组数及每轮更新次数，直接影响训练过程       |
`train_data_size`会改变：
- **每轮任务覆盖量**：任务更多，通常能减少少数任务造成的波动。
- **每轮训练样本量**：轨迹展开后，单步样本更多。
- **每轮参数更新次数**：固定 mini-batch=256 时，数据越多，通常更新次数越多。
- **重新采样的频率**：每次重新 rollout 前，要先训练完更大的一批数据。
不过它**不会改变每个任务组内的轨迹数**，组内仍然是 8 条，所以不是把单个任务的 GiGPO 比较组变大了。
**【但是】值得注意的是`train_data_size`会显著提高rollout期间CPU的工作量。因为rollout需要的workers数量就等于需要的轨迹条数：`train_data_size*group_size`。与此相关的还需要调整 `num_cpus_per_env_worker=0.1`， 代表CPU分配多少份额给Ray调度。这个值太高会导致cpu全用来给这群worker，导致其他主要进程卡住进不去训练；太低会导致可分配的RAM不够，进程直接死掉。

对你当前配置，调参时可以这样区分：**显存不够优先减 micro；想改变采样规模和训练行为，再调 `train_data_size`。** 增大 `train_data_size` 也不是必然提升效果，因为每轮的采样成本和更新量都会一起增加。

训练过程：
```
当前权重 θ₀
  ↓
4 个任务，每个任务运行 8 条轨迹，共 32 条轨迹
  ↓
把轨迹展开为单步训练样本
  ↓
分 mini-batch 训练这批样本 1 遍，得到 θ₁
  ↓
用 θ₁ 重新 rollout 下一批 32 条轨迹
  ↓
继续训练……
```
`trainer.test_freq=5` 指的是：**每完成 5 轮“rollout → 训练”，做一次验证。**
因此在wandb里每隔5个step会打印出一段rollout片段示例

ps.**“32 条轨迹”不等于“32 条训练数据”。** 每次环境交互都会产生一条“输入 prompt → 模型回答”的训练样本：
- 如果平均每条轨迹走 30 步：约 `32 × 30 = 960` 条训练样本。
- 全部走满 50 步：最多 `32 × 50 = 1600` 条训练样本。
- 为满足并行整除要求，代码可能再复制少量样本补齐。
**实际 optimizer 更新次数由样本数决定。** 当前全局 mini-batch 为 256，`ppo_epochs=1`，因此每轮大致为：
```
optimizer 更新次数 = ceil(展开后的样本数 / 256)  #（整除）
```
例如：

|本轮展开后的样本数|optimizer 更新次数|
|---|---|
|960|4|
|1280|5|
|1600|7|


# 主要报错以及对应原因
## CUDA OOM
爆显存了，需要调低`ppo_micro_batch_size_per_gpu`和`log_ppo_micro_batch_size_per_gpu`

## OOM(RAM)
内存不够，会报OOM ERROR并结束作业任务，节点断开
这部分往往是出现在rollout阶段，一堆env worker开始工作的时候（而不是真正的训练阶段），一方面是`num_cpus_per_env_worker`给的预留比例太小，导致预留的内存不足，可以调大`num_cpus_per_env_worker`(调大会有别的问题，见下文) ；另一方面是`train_data_size*group_size`太大了，可以适度调低。（虽然我不想调低这个）

可以通过该命令查看GPU、CPU资源分配
```
scontrol show job 55119 #(换为具体的JOBID)
```

申请时通过`--mem=256G`参数申请更多的内存（原来默认好像是6G*16核心=96G）
```
sbatch -N 1 --gres=gpu:2 --mem=256G -p project3-1 -A project3 scripts/train_gigpo.sh
```

## 训练前卡住，终端无输出，GPU不工作
是因为`num_cpus_per_env_worker*train_data_size*group_size`（也就是总的rollout-worker数乘每个worker占用的CPU单位）太多了，加上GPU、task调度线程总数超过了128个（Ray默认分配的虚拟CPU核数），导致后面的计算任务卡住。
如何查看：
在**作业所在的计算节点**激活 NEST 环境后执行：
```
conda activate NEST
ray status
```
看输出中的：
```
Resources
---------------------------------------------------------------
Total Usage:
 28.6/128.0 CPU
```
这里：
- `128.0`：Ray 初始化的 CPU 总调度额度。
- `28.6`：当前已经预留的额度，**不是实际 CPU 利用率**。
遇到卡住时，调小上述值可以解决



# 训练日志
1. zyt115200-harbin-institute-of-technology/nest/3l5dyfr7  GiGPO，Qwen2.5-1.5B-Instruct第一版完整训练