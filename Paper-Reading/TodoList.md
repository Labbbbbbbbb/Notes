HEPO
RLHF
DPO
Reject-Sampling
PRM

TreeRL Hou et al. (2025) integrates an entropy-guided sampler (EPTree) that branches at uncertain tokens, then back-propagates leaf rewards to provide global and local (step) advantages thereby eliminating a separate process reward model.

Tree-OPO Huang et al. (2025) leverages off-policy teacher MCTS to build prefix trees and proposes staged, prefix-conditioned advantage estimation to stabilize GRPO-style updates.

寻找benchmark，做最小可行性验证

[Qwen3 Technical Report](https://arxiv.org/html/2505.09388.pdf#abstract1) ....呃？（重点看一下post-training部分）
OPD
MoE
Q:目前的post-training算法大都只取模型的输出而不细究模型是dense/MoE架构，有没有一种后训练方法会针对MoE的路由来进一步分配奖励呢？
A:有的——而且这正是 2025 年底到 2026 年刚兴起的一条明确研究线。不过先给一个重要的定性判断： 现有工作几乎都把路由信息用于「稳定训练 / 负载均衡 / 探索」，真正做「哪个专家导致了成功」的反事实专家级信用分配的方法基本还没有 ——后者恰好是开放空白。下面按技术路线分四类介绍。
1,RO-GRPO （Routing-Optimized GRPO）是最字面意义上「针对 MoE 路由机制分配奖励」的方法
2,RSPO （Router-Shift Policy Optimization，北大+微软，2025.10）不改奖励，而是 按路由状态给每个 token 的策略梯度重新定权 ——本质上是「路由条件化的 token 级信用分配」
3,ESRL（Expert-Space Exploration RL） ：rollout 时对 router logits 注入自适应噪声做专家空间探索，配合 anchored expert sampling （锚定部分专家保证质量）和 rollout routing replay （训练时重放 rollout 的路由决策）更新；在多款 MoE 上验证数学/代码/GPQA 收益，并观察专家负载变化
4,搜索中还会出现 DMoERM 、PrefMoE 这类工作——它们是 把奖励模型本身做成 MoE 结构 （按任务/标注者风格路由到不同奖励专家），改进的是奖励信号的质量， 并不根据 policy MoE 的路由去分配信用 。和你问的问题是两个方向，别混。