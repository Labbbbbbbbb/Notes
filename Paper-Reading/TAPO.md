Related work部分小记
**1. TPPO**（微软，2024） Token-level PPO，用在**搜索引擎查询词生成**上：训练一个 token 级奖励模型（靠大模型给每个 token 打相关性分），再做逐 token PPO。问题：生成序列极短（≤10 token），且**依赖一个大型辅助模型做 token 奖励标注**——在 ≥2K token、含大量 CoT 的多轮轨迹里，这种标注根本不可行。

**2. POAD**（上海交大等，NeurIPS 2024） Policy Optimization with Action Decomposition：把“一个动作 = 一串 token”这件事正式建模，推导 Bellman backup 同时给动作内 token（intra-action）和动作间 token（inter-action）分配信用。问题：验证场景是**短程、无 CoT 推理**的简单环境，而复杂环境里去掉 CoT 性能会显著退化。

**3. ARPO**（人大/快手，ICLR 2026） Agentic Reinforced Policy Optimization：观察到 **LLM 每次收到工具返回结果后，接下来几个 token 的熵会飙升**（不确定性被工具结果重新引入），于是在这些高熵步骤做**自适应分支采样**——rollout 阶段在分岔点多采几条路径。注意 TAPO 的措辞：ARPO 的高熵意识用在**采样/探索阶段**，而不是训练时的梯度更新上；且它面向工具集成推理（搜索类任务），不是 ALFWorld 这类原生交互环境。

**4. 熵坍塌研究与 AEnt**

- Cui 等（2025）《The entropy mechanism of RL for reasoning LMs》发现 RL 训练中策略熵会过早坍塌（模型过早锁定答案、丧失探索），是 LLM RL 的关键障碍。
- **AEnt** 等熵感知方法据此在训练中主动维持/调控熵，鼓励持续探索。

**5. Wang et al. ——和 TAPO 最相关的工作** 清华+阿里的 _Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning_。核心发现：CoT 里只有**少数高熵 token 是“forking tokens”（分岔 token）**，决定推理走向；**只对这部分 token 做梯度更新，效果就能持平甚至超过全量更新**（32B 模型上反而涨 7~11 分），而只训低熵 token 则明显掉分。

**TAPO 的定位**：它把  的“只更新高熵 forking token”从**单轮数学推理**搬到多轮交互环境，并补上对方缺的另一半——**token 级的正负 advantage（从整轨迹终局回报直接算，含失败惩罚）**。两者结合 = 熵筛选关键 token × 每个 token 带独立的正负优势，这就是 §2.3 结尾说的 "uniquely combines"。