**ARPO，LLM-based tool-use agents.，针对multi-turn场景（是ALFWorld那种multi-turn，不是CREST那种）**
从两个pilot experiment（HotpotQA，AIME2025）提出一个关于熵的机制，即每次工具调用（tool-call）之后token的熵会剧烈上升，因为每次tool-call后环境变换，导致反馈中有较大的分布偏移，由此论文提出：
>[!Note]
>**These findings highlight a limitation of trajectory-level RL methods, which focus on initial reasoning while overlooking the uncertainty introduced by tool-call feedback.**

### 1.entropy-based的rollout机制

>[!Note]
>**In the rollout phase, the LLM initially performs multiple global samplings, recording the initial entropy distribution of each sample. After each tool-calling, we further monitor the real-time token entropy variation, regarding them as branching criteria. If the entropy variation exceeds a predefined threshold, the model triggers additional partial sampling to explore alternative tool-integrated reasoning paths.**

但是注意它的trigger机制，这里的partial sampling是以step为基本单位的，看的是tool-calling之间一段reasoning tokens的entropy（而不是根据每个token的熵来决定）,详情可以去看原文的3.1，写得挺详细的
![[Pasted image 20261004011127.png]]
### 2.ADVANTAGE ATTRIBUTION ESTIMATION
探究在rollout之后如何分配advantage
其实这个机制还是挺有意思的，GiGPO面对这样的问题选择了树状的分支（是叭？和TEMPO的思路一样），同时可以注意到GiGPO和ARPO面对的rollout结果其实很相似，都是前缀分支加分叉，不一样的是GiGPO是一股脑rollout完了再统计哪些共享前缀哪些是分叉点，而ARPO是主动地、根据熵来主动选择分叉点

此处的处理分为Hard和Soft两种，Hard是如论文中所写，显式地处理了分支A和共享A的表达式，Soft则是借助rollout的机制和GRPO的更新机制，说明在GRPO的基础上不需要特别处理A也可以使其估计产生和Hard一致的理论效果。且实验中soft表现更好。

### 3.理论分析
![[Pasted image 20261004021109.png]]
其实这个结论还挺惊艳的，直接把LLM agent的动作优化和传统策略优化结合起来了。
但是结合token级别的背景来看可能会有一些难理解的地方：这并不代表同样动作中的每一个不同的token在整轮训练中都会获得一样的梯度，对吧
![[Pasted image 20261004021336.png]]
关键在于在这个场景中，所谓的“一样”的动作$A_T$，其实可以因为token的细节不同而从**非**语义的角度有狠毒偶不同，因此虽然上述的等效从理论上很好地优化了step-level的策略，但是语义相同的策略在token-level上的不同以及这部分不同是否影响这个step的生成仍是有探索空间的。