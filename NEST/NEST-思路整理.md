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

\Delta _{ A_ {\text{step}}} = \mathbb{E}[A^{\text{step}} \mid \text{ Success }] - \mathbb{E}[A^{\text{step}} \mid \text{ Failure }]

$$
衡量Token-level的区分是有益还是冗余:
$$
D_i = \frac{ 1 }{T_i} \sum_{t= 1 }^{T_i} \frac{| A_{i,t} - A_i |}{| A_i | + \epsilon}
$$
（3）两个粒度的置换实验，打乱其中一个粒度的优势估计，分析其成功率的变化，若成功率明显降低，说明该粒度的优势估计有意义，否则说明冗余。


## ToDo

1.跑一下VeRL，熟悉一下WebShop和ALFWorld，做最小可行性验证

## 想法随写
tips1：TEMPO和TAPO对token-level的区分虽然方法不同但是是存在一些关系的，TAPO的Shannon熵正式确定性不高的toekn，而正是在这样的token上会产生较多分支，两种方法最后都在让模型重点更新不确定的token点并使其往高advantage的方向走
