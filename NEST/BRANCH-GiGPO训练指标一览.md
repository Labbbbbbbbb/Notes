按当前的 train_gigpo.sh 来看，默认运行的是 **ALFWorld + GiGPO**，W&B 项目名为 `nest`，实验名为 `gigpo_alfworld`。
**1. 任务表现：最值得关注的一组**

| 指标                             | 含义                                                |
| ------------------------------ | ------------------------------------------------- |
| `episode/success_rate`         | 当前训练批次中，完整任务的成功比例。`0.5` 表示 50% 的轨迹成功              |
| `episode/{任务类型}_success_rate`  | 某一类任务的成功率，例如拿取并放置、加热后放置等                          |
| `episode/reward/mean、max、min`  | 完整轨迹累计环境奖励的平均值、最大值、最小值，不包含训练器后来追加的无效动作惩罚          |
| `episode/length/mean、max、min`  | 一条完整轨迹执行的环境步数。当前最多 50 步                           |
| `episode/valid_action_ratio`   | 环境判断为有效的动作比例。有效不代表能完成任务                           |
| `episode/tool_call_count/mean` | 每条轨迹的工具调用次数；普通 ALFWorld 环境动作不一定计入这个字段，出现 0 属于正常情况 |

当前文本版 ALFWorld 的环境奖励是 `10 × won`。因此，完整轨迹奖励主要反映是否完成任务。

任务类型包括：

|名称片段|任务含义|
|---|---|
|`pick_and_place`|拿取物体并放置|
|`pick_two_obj_and_place`|拿取两个物体并放置|
|`look_at_obj_in_light`|在灯光下查看物体|
|`pick_heat_then_place_in_recep`|加热后放入容器|
|`pick_cool_then_place_in_recep`|冷却后放入容器|
|`pick_clean_then_place_in_recep`|清洁后放入容器|

**2. 验证集表现：判断泛化效果**
当前配置在训练前验证一次，之后每 **5 个训练 step** 验证一次，最后一步也会验证。

|指标|含义|
|---|---|
|`val/success_rate`|验证环境中的整体任务成功率，建议作为主要效果指标|
|`val/{任务类型}_success_rate`|验证环境中各类任务的成功率|
|`val/text/test_score`|数据源为 `text` 的验证样本平均评分。这里按展开后的环境 step 行统计，**不是完整轨迹成功率**|
|`val/text/tool_call_count/mean`|验证轨迹的平均工具调用次数，按轨迹去重后计算|

这里 `text` 是数据源标签。

训练时每个任务采样 8 （group_size）条轨迹，而当前验证任务数设置为 8（val_data_size），验证也使用随机采样。因此成功率可能明显跳动，需要结合多次验证看趋势。

实现见 ray_trainer.py (line 849)

**3. `critic/*`：奖励和优势统计**

**虽然名字带 `critic`，当前 GiGPO 路径并没有训练价值网络。** 这些名称沿用了 PPO 框架的日志命名。

|指标|含义|
|---|---|
|`critic/score/mean、max、min`|每个训练行的 token 分数之和，再统计均值和极值。这里一行对应一次环境交互|
|`critic/rewards/mean、max、min`|最终用于优势计算的每行奖励之和|
|`critic/advantages/mean、max、min`|有效回答 token 上的优势统计。正值表示应提高相应行为的概率，负值表示应降低|
|`critic/returns/mean、max、min`|当前 GiGPO 实现直接把优势同时写入 `returns`，所以它和 advantages 相同，并不是原始累计回报|

有两个容易误读的地方：

- **`score` 已经包含无效动作惩罚。** 当前无效动作扣 `0.1`，所以能看到 `-0.1`。
- 当前 `algorithm.use_kl_in_reward=False`，因此 `score` 和 `rewards` 相同。KL 在模型损失中施加。

GiGPO 的优势由两部分组成：
```
最终优势 = 轨迹组内相对优势 + step_advantage_w × 状态组内相对优势
```
当前 `step_advantage_w=1.0`。优势是相对学习信号，均值接近 0 不代表训练没有效果。
指标统计见 metric_utils.py (line 89)，优势计算见 core_gigpo.py (line 163)

**4. `actor/*`：模型更新是否稳定**

| 指标                        | 含义及怎么看                                            |
| ------------------------- | ------------------------------------------------- |
| `actor/pg_loss`           | 策略梯度损失。用于优化行为概率；可以为负，不能像监督学习 loss 那样单纯追求持续下降      |
| `actor/entropy_loss`      | 更新前策略在有效回答 token 上的平均熵，反映输出分布的不确定性。高表示更多样，低表示更确定  |
| `actor/kl_loss`           | 当前策略相对参考模型的 KL 惩罚估计，反映偏离初始参考策略的程度                 |
| `actor/kl_coef`           | KL 损失权重，当前为 `0.01`                                |
| `actor/ppo_kl`            | 当前更新策略相对本轮采样旧策略的近似 KL。它和参考模型 KL 是两回事，采样估计可以略为负    |
| `actor/pg_clipfrac`       | PPO 裁剪实际约束到目标的有效 token 比例；过高意味着很多更新被裁剪            |
| `actor/pg_clipfrac_lower` | 负优势情况下触发 dual-clip 额外约束的比例，不是一般意义上的“概率比低于下界比例”    |
| `actor/grad_norm`         | 梯度范数，帮助判断更新幅度和异常尖峰；这里通常记录裁剪前范数                    |
| `actor/lr`                | 学习率，配置初始值为 `1e-6`；控制台保留三位小数时可能显示 `0.000`，并不代表实际为零 |

当前 actor 优化目标大致是：
```
总损失 = pg_loss − 0.001 × entropy + 0.01 × kl_loss
```
W&B 没有单独记录这个完整总损失，而是记录上述分量。
实现见 dp_actor.py (line 390)

**5. 输入和输出长度**

|指标|含义|
|---|---|
|`prompt_length/mean、max、min`|每次动作生成时，输入 prompt 的有效 token 数|
|`response_length/mean、max、min`|每次生成回答的有效 token 数|
|`prompt_length/clip_ratio`|输入长度等于当前张量容量上限的比例|
|`response_length/clip_ratio`|回答长度等于当前张量容量上限的比例|

当前配置输入最多 2048 tokens，回答最多 512 tokens。这里的长度是**单次环境交互**的文本长度，轨迹长度 `episode/length` 的单位则是环境步数。
`clip_ratio` 偏高提示可能频繁顶到长度上限，但该实现只判断长度相等，没有检查具体结束原因。

**6. 生成与训练概率的一致性**

|指标|含义|
|---|---|
|`training/rollout_probs_diff_mean`|生成后端与训练后端对已生成 token 计算出的概率绝对差的平均值|
|`training/rollout_probs_diff_max`|上述概率差的最大值|
|`training/rollout_probs_diff_std`|上述概率差的标准差|

这里比较的是 `exp(log_prob)` 后的**概率差**。用于观察 vLLM 生成与训练端重新计算之间的数值差异，通常越小越一致。

**7. 时间和资源开销**

|指标|含义|
|---|---|
|`timing_s/gen`|整个多轮环境交互耗时，包括生成和环境执行|
|`timing_s/reward`|训练奖励计算耗时|
|`timing_s/old_log_prob`|重新计算旧策略 token 概率耗时|
|`timing_s/ref`|参考模型计算概率耗时|
|`timing_s/adv`|优势计算耗时|
|`timing_s/update_actor`|actor 参数更新耗时|
|`timing_s/testing`|验证耗时，仅验证轮出现|
|`timing_s/step`|整个训练迭代耗时，包含本轮验证等操作|
|`timing_per_token_ms/*`|对应阶段按 token 数归一化的耗时。`gen` 除以回答 token 数，其他已记录阶段除以输入加回答 token 数|
|`perf/total_num_tokens`|本轮训练批次的输入加回答有效 token 总数|
|`perf/time_per_step`|与 `timing_s/step` 相同|
|`perf/throughput`|`总 token 数 ÷ step 秒数 ÷ GPU 数`，单位为 tokens/s/GPU|
|`perf/mfu/actor`|根据估算计算量得到的模型算力利用率，不等同于 `nvidia-smi` 的 GPU 利用率|
|`perf/max_memory_allocated_gb`|PyTorch 记录的峰值已分配显存|
|`perf/max_memory_reserved_gb`|PyTorch 缓存分配器记录的峰值预留显存|
|`perf/cpu_memory_used_gb`|worker 所在主机的系统已用内存，不是当前 Python 进程独占内存|

显存指标未在每轮重置峰值，因此不能直接当成本轮显存峰值使用；worker 返回的指标还会经过聚合。

**8. 并行负载和训练进度**

|指标|含义|
|---|---|
|`global_seqlen/min、max`|均衡前，各数据并行分区的 token 总量最小值、最大值|
|`global_seqlen/minmax_diff`|均衡前最大与最小分区 token 总量的差|
|`global_seqlen/balanced_min、balanced_max`|重新分配样本后，各分区 token 总量的最小值、最大值；越接近说明负载越均衡|
|`global_seqlen/mean`|每个分区平均分配的 token 数|
|`training/global_step`|完成到第几次训练迭代，不是环境交互次数，也不是内部优化器更新次数|
|`training/epoch`|训练数据遍历轮数，从 0 开始|

此外，W&B 会收到完整训练配置，初始化代码在 tracking.py (line 45)。项目还支持 `val/generations` 验证文本表格，但当前 `log_val_generations=0`，默认不上传。W&B 页面中的 CPU/GPU 系统监控则由 SDK 根据运行环境自动采集。

实际看实验时，建议优先固定这几条曲线：**`val/success_rate`、`episode/success_rate`、`episode/valid_action_ratio`、`actor/kl_loss`、`actor/entropy_loss`、`actor/grad_norm` 和 `perf/time_per_step`**。它们分别回答任务有没有学会、动作是否有效、更新是否稳定，以及训练成本是多少。