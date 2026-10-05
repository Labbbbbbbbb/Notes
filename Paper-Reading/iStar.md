-AGENTIC REINFORCEMENT LEARNING WITH IMPLICIT STEP REWARDS，ICLR2026
-**“How can we design a credit-assignment strategy that is label-efficient and stable, scales to multi-turn interactions, and remains robust and generalizable to (un)verifiable rewards in open-ended environments?”**
-step-level，PRM

用multi-turn DPO的思想训练了一个implicit PRM来提供step-level的reward并计算advantage，最后的优势是episode-level advantage和step-level advantage的加法调和。

并且这篇是特地说了只做step-level不做token-level的PRM，因为噪声大等等

**后面可以关注一下vanilla DPO和本文所说的这个multi-turn 的DPO有什么联系**
![[Pasted image 20261004191312.png]]
有意思的点是Multi-turn DPO中的$r_{i,t}$的分母是上一次更新前的快照$\pi_{old}$，而非DPO中的$\pi_{SFT}$（已冻结），并且所有步级的$r_{i,t}$按时间步t累加起来，log相加等于里面的值相乘，最后结果相当于log下新模型$\pi_\phi$对这一串轨迹的喜好比上$\pi_{old}$产生这一串轨迹的概率
想证明的核心是：**虽然 DPO 损失只在整条轨迹层面优化（比较 τ1​ vs τ2​），但数学上这个最优解 πϕ∗​ 隐式定义了一个合法的、可分解的步级奖励函数。**
![[Pasted image 20261004192628.png]]


>[!NOTE] 一些值得注意的观点（ps.不是这篇文章的主要内容，只是觉得这个观点有必要记一下）
>**trajectories are long and non-Markovian in token level, with each step consisting of a chain-of-thought (CoT) (Wei et al., 2022) and an executable action, inflating variance when credit is pushed to individual tokens;**
>- **non-Markovian（非马尔可夫）**：在 **step 层面**环境是近似马尔可夫的（st+1​ 由 st​ 和动作决定，TextWorld 这类环境确实如此）；但在 **token 层面不是**。一个动作（如 `"take hotdog 1"`）由一长串 CoT token + 动作文本组成，第 1 个 token 生成时，它"所处的状态"依赖于之后整段生成的内部演化——你无法定义一个 token 级状态转移函数 P(st+1​∣st​,at​)，马尔可夫性是 PG 定理干净成立、价值函数可稳定估计的基础。
>- **"硬把信用推到 token 级会导致策略梯度估计的方差爆炸，实践上不可靠"**
>
> **Implicit PRMs (Yuan et al., 2025; Cui et al., 2025) help in single-turn tasks, but the token-level process rewards tend to be overly fine-grained in agent training, amplifying variance and destabilizing training as trajectories grow.**





