**立意（参考任务书）：**
多轮大语言模型智能体（multi-turn LLM agent）的强化学习后训练中， 信用分配粒度是决定训练效率与最终性能的关键因素。当前主流方法各自在单一粒度上做优势估计，一是序列级如GRPO，以完整轨迹为单元计算组内相对优势，信号最粗，无法定位轨迹内部哪一步、哪个词元贡献了奖励；二是步骤级如GiGPO，以agent的动作步为单元，通过相同环境观测的轨迹分组计算步级相对优势，但步内部仍无区分；三是词元级如TAPO的熵过滤，利用策略熵识别关键推理token，能定位"哪个词元是决策关键"，但缺乏对多轮交互结构的建模。

项目核心目的在于讨论步骤级与词元级两个粒度的信用分配是否能正交组合产生叠加增益，目前文献中尚无工作系统性地回答这一问题，GiGPO和TAPO/TEMPO分别独立提出并验证了各自的有效性，但从未在统一框架下做对照消融，更未探索二者的乘法调制能否优于任一单独方法。因此本课题提出NEST（Nested Estimation of Step and Token advantages），旨在通过实证的方式探究两个粒度之间是否正交/加成/冗余，并实现一种轻量化的嵌套优势估计方式。

**整理了以下几点需要做的事情：**
（1）步骤级信用分配（GiGPO anchor grouping）与词元级信用分配（TAPO熵过滤/TEMPO前缀树）在多轮agent训练中是否存在正交互作用，并通过置换实验验证两个粒度分别的作用。
（2）不同嵌套方式，如乘法调制($A^{step}⋅w^{token}$)和加法调制(类似TEMPO)，是否存在效果差异。
（3）使用VeRL框架实现一个轻量化的后训练强化学习优势嵌套估计范式。

**主要技术指标：**
（1）在ALFWorld、WebShop等benchmark上，完整NEST相比最强单一粒度方法成功率点的提升。
（2）Step-level与Token-level的区分度
衡量Step-level的优势能否区分成败轨迹:
$$

\Delta _{ A_ {\text{step}}} = \frac{\mathbb{E}[A^{\text{step}} \mid \text{ Success }] - \mathbb{E}[A^{\text{step}} \mid \text{ Failure }]}{\mathbb{E}[A^{\text{step}} \mid \text{ Success }] + \mathbb{E}[A^{\text{step}} \mid \text{ Failure }]}

$$
衡量Token-level的区分是有益还是冗余: （其中$A_i$是第i步的优势，$A_{i,t}$是第t个token的优势。
$$
D_i = \frac{ 1 }{T_i} \sum_{t= 1 }^{T_i} \frac{| A_{i,t} - A_i |}{| A_i | + \epsilon}
$$

**注意：** 这一点的数值要求（要求两个区分度分别大于xxx）其实只是统计意义上的指标，只能说明数据质量而不能说明训练出来的模型质量，但是这个指标是由我们设计出的算法计算出的，并且最后直接作为策略的梯度，因此对这个指标进行要求可以算是对实验原理的要求。但是要注意一个问题是Advantage是有正负的，相比之下价值V一般没有符号区别，因此定指标时要搞清楚其意义时价值还是优势
。。但是总觉得有点颠倒啊，应该是算出来的梯度是正是负来决定动作概率是增加还是减少，而不应该是预设一个”好“/”坏“的token或者step来要求它的梯度啊，本来就是一个统计意义的......

遂把这一点改成---->
（2）两个粒度的消融实验，为说明该粒度的优势估计有意义，去掉其中一层的信息后，其成功率应降低5%以上。

（3）两个粒度的置换实验，打乱其中一个粒度的优势估计，分析其成功率的变化，若成功率明显降低，说明该粒度的优势估计有意义，否则说明冗余。


。。等等等等。先想清楚这个优势估计如何”嵌套“，是需要为每个step和token分别有一个计算优势的式子，还是说想TEMPO那样，是两个优势加和（或者乘积之类的）？policy本身是用于生成token的，这样子的话似乎所有优势最终都需要汇总成token的梯度，单单说step的优势是没有落点的。除非在agent中另外训练一个专门预测step的特征向量（representation），把这个向量作为一个生成step中的每一个token的条件，这样子的话才有可能真的去估计step的优势同时它真的能作为一个梯度去优化某个东西。我觉得这是一个很有价值的想法（吗），至少它真的符合嵌套的思想。
![[Pasted image 20260921222410.png]]
看一下这两篇的想法
 ① CREST（[arXiv:2608.13179](https://arxiv.org/abs/2608.13179) ，2026.08）——Agent 场景，两层嵌套
 同时尽管拿teacher来当magnitude的方法有效避免了错误方向和reward hacking，但是一个turn失败就代表所有token均应该背离teacher的做法，也不尽合理，P1 性质 sign(At​)=sign(Aturn) 既是它的安全保证，也是它的表达力上限。
 ② SHAPE（[arXiv:2604.06636](https://arxiv.org/html/2604.06636v1) ，华为/北大，2026.04）——单轮推理，两段式
## ToDo

1.跑一下VeRL，熟悉一下WebShop和ALFWorld，做最小可行性验证
2.整理参考文献的benchmark和所用模型大小、训练框架，包括GiGPO、TEMPO、DAPO、HEPO、TAPO、OPD、OPSD、CREST、有时间的话包括RLHF、RLVR等
这个整理包括方法本身是agentic RL还是reasoning RL 、benchmark给出的是什么粒度的reward（traj or step）

## 想法随写
tips1：TEMPO和TAPO对token-level的区分虽然方法不同但是是存在一些关系的，TAPO的Shannon熵正式确定性不高的toekn，而正是在这样的token上会产生较多分支，两种方法最后都在让模型重点更新不确定的token点并使其往高advantage的方向走

tips2：‘training-time methods that respect the turn structure of multi-turn sessions.’这个概念很有意思，出现在CREST的related work，感觉有点像multi-turn OPD

tips3：More recently, hybrid methods attempt to combine RL’s verifier-bounded direction with distillation’s dense signal: SDAR (Lu et al., 2026b) gates self-distillation as an auxiliary loss alongside RL（我感觉CREST的related work好多可以学的地方、、）

tips4：HEPO主要是研究RLVR关于熵的动力学，指出只给高熵token梯度可以抑制熵坍缩和探索坍缩，这方面似乎和DAPO的机制有异曲同工之处，可以进一步思考下。于与此同时还有SFT和RL阶段对于熵的处理不同导致的“SFT死记但RL能泛化”，也可以体会一下。基于熵和基于概率的筛选思想有所不同，具体可以看一下Relative Surprisal Index。

tips5：如果说高熵掩膜是为了防止低熵token倾向于坍缩（例如90%的可能采样到一个token并且它是好的，就会无限继续提高它的可能性，鉴于低熵本就是由已有的语言知识习得的，这是大概率的事情），那么是不是熵作为门控而非乘在advantage上可以更好地防止熵快速坍塌？
![[Pasted image 20260929020314.png]]

tips6:Clip-higher 放宽正 advantage token 的 ratio 上界,难道不是鼓励正确的top-k变得更自信，难道不是会让熵更容易变小吗
针对这个疑惑的关键事实是**放宽的上界对 top token 根本不可达**
![[Pasted image 20260929015430.png]]
group 内正 advantage 落在"挑战分支"上,不是落在 top token 上，因此DAPO鼓励高熵选择中的低概率token概率变高，倾向于更高熵地探索
## 构想
### 1.开题意义部分，可以参考一下CREST，写一下RLVR
single-turn-->multi-turn
RLVR
Entropy-perspective
**从entropy的视角看---token级熵掩膜的原因**
“This balance is especially fragile in RLVR for large models. When entropy is too low, the policy converges prematurely to suboptimal behaviors (entropy collapse); when it is too high, uncontrolled stochasticity attenuates learning signals (entropy explosion). Navigating this entropy dilemma is therefore pivotal for scaling RLVR.”
DAPO通过clip-higher来避免entropy collapse，HEPO则通过截断低熵token的梯度来避免entropy collapse；QAE看到了entropy collapse和explosion两端的弊病，通过将GRPO优势构造中减去的中位数换成k-percentile，从而避免稀疏奖励中大量的negative advantage造成entropy explosion，并且可以使得大部分token的优势为0（咋做到的）。
以上的三种方法（DAPO、HEPO、QAE）都是单轮的reasoning RL，其中后两者都在AIME24/25上进行了测试，可以借助is_equivalent函数高效判断出是否获得奖励

**从别的视角看---alternative token-level indicators**
Do not let lowprobability tokens over-dominate in rl for llms. In 2nd AI for Math Workshop@ ICML, 2025.
用RSI指标来进行token-level筛选
[4] Xingwu Chen, Tianle Li, and Difan Zou. Reshaping reasoning in llms: A theoretical analysis of rl training dynamics through pattern selection. arXiv preprint arXiv:2506.04695, 2025.
[9] Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. arXiv preprint arXiv:2507.00432, 2025.

**Recalibrating token contributions--调整权重**
（简单搜一下讲了什么就好）
[3] Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346, 2025.
[33] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Edge-grpo: Entropy-driven grpo with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025.
[23] Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, and Yanyong Zhang. On the entropy dynamics in reinforcement fine-tuning of large language models. arXiv preprint arXiv:2602.03392, 2026.
⁨[30] Jiarui Yao, Ruida Wang, et al. Future-kl regularized grpo: Process-level credit assignment from f-divergence regularization. arXiv preprint arXiv:2601.10201, 2026.

**On-Policy Distillation**
benchmark为XSum（输入文章生成摘要），WMT（英德语翻译）,GSM8K（小学数学CoT推理看），其实本身是一个知识蒸馏（KD）的优化算法，不直接属于RLVR，但是值得注意的是这个on-policy的范式让它可以丝滑地与RLVR相结合，并且文中给出了融合RL奖励与KL散度的优化目标

**TPPO**

**LLM Agent相关工作**
**TAPO**：

**GiGPO**：
**TEMPO**：
**GRPO**和它的varient（主要是DAPO）

**CREST**的机制是由环境的turn-level Reward得到优势的正负，然后利用 privileged self-teacher，并通过 entropy-gated modulation 调节这个 advantage 在不同 token 上的幅度（credit magnitude），最终得到 token-level credit。
所以粒度上：**session > turn > step > token**（**CREST / BFCL / WildToolBench 这类多轮会话场景**）：turn 和 step 严格区分如上（一个 turn 内含多步工具链）。。CREST 的两层信用分配正好对应：
- inter-turn：解决"哪个 turn 成功/失败"（环境只需在 turn 边界给 Rk​）；
- intra-turn：解决"这个 turn 内哪个 token 更关键"（self-teacher+entropy-gate 负责，环境不参与，并且计算magnitude的粒度是直接到token级别的）。
呃这个啥比说法，我一直以为turn就是step的。。。但是尽管如此，teacher只用于调整magnitude尽管能避免reward hacking，但是仍会限制其表达能力（只要最终结果是好的，每个token都会是正梯度，不管其中是不是有暗藏的坏token）；并且在本文的语境下，其实env只提供了trajectory-level的ground truth奖励，然后利用OPSD和entropy直接给出了token-level的credit，其实可以说并没有一个嵌套的思想。
![[Pasted image 20261003205545.png]]

GLAM和TWOSOME：预设动作集（而不是生成任意自然语言token来和任务做语义匹配），然后用RL进行优化。每一条轨迹由简单的`turn left、turn right、go forward、pick up、drop、toggle`构成，优化的粒度是action级，而非token/trajectory level。action的优势基于critic计算给出，而action的概率由每个token生成概率的乘积得到

**POAD**：（可以去看一下POAD那章的小结），在上面两篇工作的基础上不再预设动作集（anyway，似乎这个action的长度还是明显较短的），并且进行了token粒度的creadit assignment，探究了稀疏奖励分配中衰减系数应如何处理才能使得intra-action和inter-action之间的价值估计始终保持一致。本质上是直接由稀疏traj级奖励直接训练token-level的价值模型（且用BAD处理系数），再用PPO进行优化的方法，没有动用熵或者筛选膜对单个token特别处理

GLAM、TWOSOME、POAD的benchmark和所用的模型：


**ARPO**：agentic RL，针对multi-turn场景（是ALFWorld那种multi-turn，不是CREST那种）
使用了HotPotQA(多跳推理，模型必须**多轮调用检索工具**（第一跳的结果决定第二跳查什么)，AIME2025(针对reasoning能力)（**AIME** 在这个 pilot 里**不是"无工具的纯推理体检"**，而是以 **TIR（Tool-Integrated Reasoning）** 的形态出现的：解数学题时模型可以调 Python 解释器执行代码（算大数、枚举、验证中间结果），代码执行结果作为 observation 回填，然后继续推理。**)

**iStar**：agentic RL，step-level，用multi-turn DPO的思想训练了一个implicit PRM来提供step-level的reward并计算advantage。
benchmark是WebShop 、 VisualSokoban and SOTOPIA,

**ReAct**：ALFWorld and WebShop、HotpotQA、Fever

**ArCHer**：agentic RL，分层学习，值得重点分析。见ArCHer笔记，benchmark ：Twenty Questions Subset, Twenty Questions, Guess My City, and WebShop


**RLHF**


开题结束后有时间看一下todolist


### 2.Nest耦合设计

>[!NOTE]
>潜在动作空间：把整句 utterance 压缩成连续 latent 向量再在 latent 上做 TD——本质就是你说的“参数化动作”，代价是 latent 与真实语言分布的失配。详细说明一下这条路径

这条路线的核心是**把“学策略”和“生成文本”解耦**：在连续 latent 空间里做决策和 TD，用一个解码器把 latent 翻译回文本。下面按机制、为什么能解 TD 难题、以及失配出在哪三部分讲。
![[Pasted image 20261005155155.png]]

![[Pasted image 20261005155803.png]]

现在的想法时在AR LLM生成的过程中，给它外接一个小的动作头，输出一个latent vector作为动作，然后作为一个条件继续参与token的生成，从而完成嵌套的动作级别和token级别的打分。听起来和COCONUT等用 LLM 自己的隐状态当 latent的路线比较接近（上图的第3条路线），但是这个隐状态要作为token生成的条件应该往哪输入呢？一定要从prompt输入吗？这样的话要想生成一串token岂不是要重复两次前向过程
![[Pasted image 20261005170219.png]]

![[Pasted image 20261005170245.png]]

![[Pasted image 20261005172923.png]]

FeUdal Networks for Hierarchical Reinforcement Learning，**ICML2017**
Controlling Large Language Model with Latent Actions，LAMDA，**ICML 2025**
Training Large Language Models to Reason in a Continuous Latent Space，Meta，**COLM 2025**

![[Pasted image 20261006161157.png]]

![[Pasted image 20261006161144.png]]