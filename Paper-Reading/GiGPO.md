## Code Reading
GiGPO 使用主训练器，没有单独的 `gigpo_trainer.py`：
- 启动脚本：[examples/gigpo_trainer/run_alfworld.sh (line 19)](/T:/Zytisworking/verl-agent/examples/gigpo_trainer/run_alfworld.sh:19)
- Python 入口：[verl/trainer/main_ppo.py (line 29)](/T:/Zytisworking/verl-agent/verl/trainer/main_ppo.py:29)
- 训练器类：[RayPPOTrainer (line 386)](/T:/Zytisworking/verl-agent/verl/trainer/ppo/ray_trainer.py:386)
- 主训练循环：[RayPPOTrainer.fit() (line 1007)](/T:/Zytisworking/verl-agent/verl/trainer/ppo/ray_trainer.py:1007)
- GiGPO 算法计算：[gigpo/core_gigpo.py (line 141)](/T:/Zytisworking/verl-agent/gigpo/core_gigpo.py:141)

虽然训练器叫 `RayPPOTrainer`，它也支持 GRPO、RLOO、GiGPO 等方法。
GiGPO 启动脚本（.sh文件）中的关键两行是：
```
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    ...
```
第一行选中主训练入口，第二行选中 GiGPO 优势估计。

主要调用流程总结为：
```
examples/gigpo_trainer/run_alfworld.sh
  │  algorithm.adv_estimator=gigpo
  ▼
verl.trainer.main_ppo
  ▼
RayPPOTrainer.fit() 
#(最核心实现，位于verl\trainer\ppo\ray_trainer.py，class RayPPOTrainer)
  ├─ 采集环境轨迹
  ├─ compute_step_discounted_returns() → step_rewards
  ├─ 构造 token_level_rewards
  ├─ compute_advantage()
  │    └─ compute_gigpo_outcome_advantage()
  │         ├─ episode_norm_reward()
  │         ├─ build_step_group()
  │         ├─ step_norm_reward()
  │         └─ 合并两种优势
  └─ 用 advantages 更新 actor
```



agent_system\multi_turn_rollout\rollout_loop.py中是rollout的具体实现，其中`    def vanilla_multi_turn_loop(`是最主要的实现之一，里面的:
```
batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs) 基于上一轮返回obs构建新的输入prompt
```

```
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded) 模型生成了新的答复，经处理后送进env。
```