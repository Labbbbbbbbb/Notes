**“Reinforcing LLM Agents via Policy Optimization with Action Decomposition”**

前请提要：GLAM and TWOSOME两个工作，也是agengt RL in interactive environment，但是动作空间是严格限定的一个动作集，只能在最终动作集中选择动作。对于一个环境，agent获得state然后开始 **递归地** 一个个生成token，最后评估生成动作a的概率时由生成每个token的概率累乘（TWOSOME加上了对动作长度的概率归一化），并在动作集上归一化，选出概率最大的动作。于此同时训练了价值网络，评估一个action的价值，从而进行更新。

而BAD的关键在于批评朴素的token（Naive Token-Level Policy Optimization）价值评估忽略了$\gamma_w$和$\gamma_a$的discrepancies，$\gamma_w<1$引入了与原始MDP不等价的优化目标，BAD证明了只有$\gamma_w=1$时这个优化才和原MDP效果等价。

BAD（Bellman backup with Action-Decomposition）迭代公式如下：
![[Pasted image 20261003013716.png]]
其中$|a_t|$是单个动作的token个数，$j<|a_t|$代表一个动作内部的token价值迭代，此时的迭代公式长这样是因为token内部R必然为0（环境并不对单个token给出奖励），而衰减系数$\gamma_\omega=1$，根据贝尔曼(最优)方程，就长成这样了，这也代表这同一个动作内部的各token内部价值评估趋向于一致，但在整个训练流程中并不会真的一样，具体分析见下面的Q1；$j=|a_t|$则代表到动作的最后一个token了，其价值衔接下一个动作，此式的$\gamma$是指动作之间的衰减系数，不等于1.而这里的$R(s_t,a_t)$是环境给出的奖励，在很多环境中是稀疏的，在action chain的末尾才非零。

POAD就是把BAD这个价值拟合方式应用到了policy optimization中。
critic：衰减系数保证了价值函数在任意时间步都保持一致性，因此可以把intra-action和inter-action的价值拟合完全融合在一起。
![[Pasted image 20261003015344.png]]
actor：
![[Pasted image 20261003015406.png]]
收集完一批trajectories后梯度更新，A的计算用广义优势估计

主要贡献：
**(1) 首次显式诊断朴素 flatten 方案的不一致性（discrepancy 的形状、长度依赖、γw​=1 的充要性）；(2) 用 on-policy 前缀值差分（望远镜守恒的 token 优势）替代 Q-Transformer 的乐观 max Q 备份（ps："望远镜"就是数学里的望远镜求和（telescoping sum）：相邻两项之差连加时，中间项全部成对抵消，像折叠望远镜一样只剩首尾。BAD 的 token 优势 δj 恰好是这种相邻差分，所以它有一个性质——**怎么在 token 之间切分信用都行，加起来严格等于动作级总优势，一分不多一分不少。）；(3) 在 LLM 智能体上用 PPO 实例化并验证。**

```
初始化: π_φ, V_θ ← LLM ρ;  V_θ̄ ← V_θ;  D ← ∅
│
for each epoch ────────────────────────────────────────────────┐
│                                                               │
│  ① Rollout (on-policy)                                        │
│     for t = 0..T-1:                                           │
│       a_t ~ π_φ(·|o_t)            # 自回归采样整条动作          │
│       o_{t+1} ~ T(o_t, a_t)                                   │
│       D += (o_t, a_t, R(o_t,a_t), o_{t+1})                    │
│                                                               │
│  ② 对 B 中每个动作 a_t 的每个 token 位置 j:                    │
│       ┌─ j < |a_t|: v_targ = V_θ̄(o_t, w^{1:j+1})   # 句内自举 │
│       └─ j = |a_t|: v_targ = R + γ·V_θ̄(o_{t+1},∅)  # 跨动作   │
│       v     = V_θ(o_t, w^{1:j})                               │
│       Â_t^j = v_targ − v        # 前缀值差分 δ^j（或 GAE）     │
│                                                               │
│  ③ Critic: θ ← θ − α∇ E[(v − v_targ)²]                        │
│  ④ Actor : φ ← φ − α∇ E[ token级 clipped PPO(Â_t^j) ]         │
│  ⑤ Target: θ̄ ← θ                                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
→ 输出 π_φ
```

### Q1 这里，如果γω等于1的话，这个训练是否完全等价于动作级别的价值网络，因为它会对一个action中的所有token给出一样的反馈。如果是这样的话那么图中所示的算法仅仅是给这些token增加了一个衰减系数。这合理吗？？
![[Pasted image 20261002164848.png]]
你的观察有一部分成立：当 $\gamma_w=1$时，同一动作内部的 token 可以共享同一份最终回报。但这不意味着它们的价值预测、优势或参数更新完全相同，也不意味着训练完全等价于动作级方法。

**朴素 token 方法与 BAD：在我们讨论的 Bellman 更新中，关键修正确实就是把动作内部的折扣从 $\gamma_w<1$ 改为 $1$，消除不必要的衰减，但是价值网络除了看现有的状态，还看已生成的前序token，设想多组数据中，有些数据前缀相同但是某一个关键token不同导致结果不同，则会导致该token的value和组内其他token不同（尽管它们在各自的组内获得一样的reward）