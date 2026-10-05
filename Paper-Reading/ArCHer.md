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