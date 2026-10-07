the Actor-Critic Framework with a Hierarchical Structure，multi-turn agent，ICLR2024
**off-policy**

>[!NOTE] 前情提要
>First of all, multi-turn RL would require online interaction with external sources such as humans or web servers, which can be slow and expensive. Due to this, on-policy methods such as PPO Schulman et al. (2017) quickly become impractical due to their inability to reuse data from past interaction. While off-policy, or even fully offline methods circumvent this problem, these methods present other challenges: since the number of tokens accumulate with multiple turns, token-level methods Snell et al. (2023); Jaques et al. (2020) that consider individual tokens as actions suffer from extremely slow convergence over long horizons. Finally, to address long horizon issues, one can treat the entire utterance for each turn as an action Verma et al. (2022); Jang et al. (2022), but this comes at the cost of introducing an enormous, variable-length action space, presenting a challenge for off-policy methods based on Temporal Difference (TD) learning that require maximization over the action at each time step. In all, this means that there is no multi-turn RL appraoch for LLMs that is both performant and efficient.
>
>**Token-level RL methods often suffer from the curse of long horizons. Utterance-level RL methods avoid this challenge, but now face the challenge of tractably optimizing for a coherent set of tokens within RL updates.**

**Utterance-level RL**：把整轮回复当一个动作
问题在于：
1. 动作空间是组合爆炸的离散序列空间（你前面深入讨论过）：maxa​Q(s,a) 不可枚举、不可微、变长，TD 求 max 失效；【要意识到的是对于这样一个动作空间爆炸且动作变长的策略，$Q_{max}$不好求（见下文Q1），因此此前看到的绝大部分方法，像GRPO、PPO都是on-policy算法，暂未看到一个off-policy RL；然而multi-turn背景下的rollout是很贵的，因此on-policy的数据利用率低是一个影响较大的问题（虽然说为什么这篇出现在了2024而未来两年基本全是GRPO等on-policy算法的天下）】
2. **coherent set of tokens**：这一句内部的 token 必须**作为一个连贯整体**被选择和更新——语法连贯、语义一致、共同完成本轮意图。你无法像独立动作那样给句内 token 分别做贝尔曼备份(non-markov)；它们共享同一个"句子级"决策，但生成时又是逐个自回归产出的；
3. 若只用句子级标量奖励做策略梯度，句内哪些 token 该为这句话的好坏负责又变成黑箱——等于把信用分配问题从"跨轮"踢到了"句内"。
针对这一点，POAD、ARPO都曾做出过理论上的分析与回答。ArCHer的处理方法是：：

![[Pasted image 20261005204859.png]]

其中一种最简单的实例算法是：
更新utterance-level价值网络：
$$
J_Q(\theta) = \mathbb{E}_{s,a,r,s' \sim \mathcal{D}} \left[ \left( Q_\theta(s, a) - r - \gamma V_{\bar{\psi}}(s') \right)^2 \right].
$$
$$
J_V(\psi) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\phi(\cdot \mid s)} \left[ \left( V_\psi(s) - Q_{\bar{\theta}}(s, a) \right)^2 \right] \right].
$$
值得注意的是此处用了Q和V两个网络，不同于经典的Q-learning用双Q更新，并且target中的Q应该是以最优的a为输入( $(Q_\theta(s, a) - (r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a')))^2$ )，这里$V(s')=E_a[Q(s,a)]$，相当于用均值替代了最优值来进行迭代
![[Pasted image 20261005205903.png]]

![[Pasted image 20261005205941.png]]

![[Pasted image 20261005212328.png]]
(a'是现场采样的，所以整体仍然是off-policy)

更新token-level价值网络：
$$
J_\phi(\pi) = \mathbb{E}_{s_c \sim \mathcal{D},\, a_t^{1:L} \sim \pi(\cdot \mid s_c)} \left[ \sum_{i=1}^{L} A(s_c, a_t^{1:L}) \log \pi_\phi(a_t^i \mid s_c, a_t^{1:i-1}) \right].
$$
此处句内每个 token 的即时奖励都是 0；
只有 utterance 结束时，拿到一个终局奖励 = 高层的 Q(s,a)−V(s)，也就是上面的$A(s_c, a_t^{1:L})$；
这个终局奖励广播到句内所有 token

**可见本文虽然有很明显的分层思想，高低两层MDP，但是高层MDP其实仍没有单独的属于自己的动作输出，它的输出就是所有token的生成。因此其实说分层，主要是奖励信号的分层--在utterance-level训练价值网络判断每一步的价值，然后在action内部用相对稀疏的奖励进行token策略优化。真正的策略网络仍只有一个，如果嵌套能做成是可以跟这个idea错开的
除此之外，本文的做法是把utterance当成整体来估计价值，然后把这个整体价值分摊到token来评估token价值。之前整理过的其他想法包括对两个层级分别独立打分（GiGPO+TEMPO），然后对两个两个层级的分数进行一定的加权调和作为token优势，这个调和的方式是不同的，不算完全重叠。同时本文仍需要单独训练一个critic，后续发展大都采用critic-free的方式**

### Q1 上面提到introducing an enormous, variable-length action space，为什么这里的action 空间大会导致这个问题啊，明明在大部分的RL环境里动作空间本来就是连续因而无限的啊
![[Pasted image 20261005151609.png]]

**针对上面问题，能否用参数化采样？**
![[Pasted image 20261005152751.png]]
这部分的大部分方法没有看懂，后面可以来研究一下。
值得注意的时它说的那句话
>[!NOTE]
>**潜在动作空间**：把整句 utterance 压缩成连续 latent 向量再在 latent 上做 TD——本质就是你说的“参数化动作”，代价是 latent 与真实语言分布的失配。

我又没看懂了sos
![[Pasted image 20261005153518.png]]