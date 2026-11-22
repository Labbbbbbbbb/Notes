# 训练日志
### TAPO/slurm-59564.out
TAPO/outputs/tensorboard/tapo_Qwen2.5-1.5B-Instruct_20261119_115928
![[Pasted image 20261122230551.png]]
效果很差，每个任务的成功率都是噪声般的震荡，没有上升效果
**优先排查的三个原因
1. **冻结参考策略的概率比存在大梯度风险。** TAPO 用 `π当前 / π冻结参考`。对负优势 token，当这个比值很大时，`min(r*A, clip(r)*A)` 仍会选中未裁剪项，因此 `epsilon=0.2` 挡不住这一侧。第 106 步，`tapo_loss` 从通常约 `0.07` 跳到 **174.98**，裁剪前梯度范数达到 **244,258**。第 79、139、142 步也有明显尖峰。actor 确实以 `grad_clip=1.0` 做了梯度裁剪，但裁剪只能缩小梯度，极端样本仍可能主导更新方向。现有日志没有逐 token 概率比，无法定位触发尖峰的样本。另外，TAPO 分支把 `pg_clipfrac` 和 `ppo_kl` 直接填成 0，`kl_loss` 则在每个 micro-batch 被覆盖。面板上的这些值不足以判断实际裁剪率和整体 KL。 [TAPO 损失 (line 76)](/data/home/zdhs0200/NEST/TAPO/tapo/core_tapo.py:76) · [Actor 更新 (line 406)](/data/home/zdhs0200/NEST/TAPO/verl/workers/actor/dp_actor.py:406) · [第 106 步日志 (line 2275)](/data/home/zdhs0200/NEST/TAPO/slurm-59564.out:2275)
2. **micro-batch 的归一化会改变样本权重。** 当前代码先在每个 micro-batch 内对有效行求均值，再按该 micro-batch 的总行数累积梯度。两个同样有 8 行的 micro-batch，假如一个只有 1 行有效、另一个有 8 行有效，前者那一行的权重会是后者单行的 **8 倍**。TAPO、成功轨迹 CE、KL 各自的有效行还不同。再加上轨迹按回合展开，回合数多的轨迹会提供更多训练行。前者是应优先修正的归一化问题，后者是在线适配时需要明确选择和验证的轨迹权重。 [有效行均值 (line 48)](/data/home/zdhs0200/NEST/TAPO/tapo/core_tapo.py:48) · [梯度累积 (line 447)](/data/home/zdhs0200/NEST/TAPO/verl/workers/actor/dp_actor.py:447)
3. **1.5B 在这套二元奖励下得到的有效信号偏少。** `tapo/nonzero_advantage_fraction` 平均约 **28%**，这是按展开后的**回合行**统计。一个任务的 8 条轨迹全失败或全成功时，整组优势为零，成功 CE 也只用混合组里的成功轨迹。TAPO 还把同一个终局优势分给一条轨迹的所有回合。相比之下，这次 GiGPO 用了环境进度奖励、回合级优势和非法动作惩罚。这些信号在初始成功率只有约 6% 时尤其重要。现有数据支持优先查奖励稀疏和回合归因，尚不能量化它们各自的影响。 [优势计算 (line 1343)](/data/home/zdhs0200/NEST/TAPO/verl/trainer/ppo/ray_trainer.py:1343) · [ALFWorld 环境奖励 (line 50)](/data/home/zdhs0200/NEST/TAPO/agent_system/environments/env_package/alfworld/envs.py:50) · [GiGPO 实际配置 (line 4587)](/data/home/zdhs0200/NEST/GiGPO/slurm-55183.out:4587)
**下一轮怎么验证
- **先做一轮保持原目标的修正版。** 保留二元奖励和冻结参考策略，分别按每个优化 mini-batch 跨 GPU 的有效行数归一化 TAPO、CE 和 KL，并修正指标。记录负优势 token 的参考概率比分位数和最大值、每批混合组数与成功轨迹数、裁剪前梯度范数。验证用同一组任务和采样种子，先跑到 **50–60 步**看趋势。
- **如果概率比尖峰仍在，只改分母做一次对照。** 在线变体用本轮 rollout 的 `old_log_probs` 作为分母，冻结参考只用于 KL。这个实验能检验长期固定的分母是否拖累在线更新；它改变了 TAPO 目标，需要作为在线变体解读。
- **如果更新稳定但成功率仍然平台，再验证信号稀疏。** 保持二元目标，先对比 `group_size=8` 和 `16` 获得的混合组数及每单位采样成本的收益。如果允许改目标，再单独测试加入环境进度奖励。学习率、熵阈值和 KL 系数放到这些对照之后，一次只动一项。