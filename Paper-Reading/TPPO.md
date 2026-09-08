论文在引言里把"学习 token 级价值函数"列为核心设计之一：_"By learning a value function that estimates the expected future rewards at the token level, the policy can make more informed decisions..."_。它批评句子级 PPO 的理由恰恰是：_"PPO is designed for multi-step RL training where the value function is typically learned to estimate the expected future rewards at each step. However, with sentence-level rewards, the value function could not accurately capture the long-term impact of individual actions"_。

所以 TPPO 的完整链路是：

1. RM 给出**稠密的逐 token 奖励** rt​（再加逐 token KL 惩罚）；
2. **critic**（与策略共享骨干、带 value head）预测每个 token 位置的 V(st​)；
3. 用 **GAE** 算优势：δt​=rt​+γV(st+1​)−V(st​)，At​=∑l​(γλ)lδt+l​；
4. PPO clipped 目标用 At​ 更新 actor，价值损失 ∥V(st​)−R^t​∥2 更新 critic。

注意：**token 级稠密奖励的作用不是取代 critic，而是让 critic 终于学得动**——每个 token 都有监督信号，V(st​) 不用再隔着几十个 token 去猜句尾那个稀疏奖励。advantage 仍然是"奖励 + 价值自举"的 GAE 组合，而非 reward 本身。