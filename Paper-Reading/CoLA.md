Controlling Large Language Model with Latent Actions，LAMDA，**ICML 2025**
任务有单轮有多轮，包括ALFWorld，但不是针对multi-turn的算法，只是通用LLM RL
>[!NOTE]
>LLM RL的效果依赖于主要组件的构建，如state、action、reward、transition等等。文本空间中状态和奖励基本由benchmark给出，然而 **the design of actions and transitions remains highly flexible and open to optimization**
>**“a framework that reformulates the language model as a transition model augmented with additional inputs of latent actions.”**


## Method
训练过程详见chapter3.2，下面简介

本文中状态$s_t$是t时刻之前输出的token，$s_{t}=x_{1:t}$，策略$\pi$负责输出latent action，环境模型transition是latent action结合现有token生成下一个token
language world model $f_{world}$：$x_{t+1}=f_{world}(x_{1:t},a_t)$ 
inverse dynamic model $f_{inverse}$：$a_t = f_{\mathrm{inverse}}(x_{1:t}, x_{t+1})$

#### **训练过程**
##### construct the latent action space and underlying world model as the basic decision modules
用一个很像VAE的方式，用$f_{inverse}$（Encoder）给出的$a_t$给$f_{world}$（Conditional Decoder，action作为条件输入）算$x_{t+1}$，来计算token预测损失，得到一个$a_t$的unsupervised学习方法

##### initialize the policy model via action-level behavior cloning
用第一步学到的Encoder $f_{inverse}$生成$a_t$，去给$\pi_\theta$学习

##### Latent Action Reinforcement Learning
冻结$f_{world}$，在环境中优化policy。（注意此时已经不再用到$f_{inverse}$，它只作为一个训练前期辅助）

##### World Model Fine-tuning under Latent Action
![[Pasted image 20261008022332.png]]


>[!NOTE] 需要注意两个问题
>1.**这里的latent action不是自由的连续向量，而是从候选动作向量中选出的离散选择**，”For the latent action space design, we employ discrete latent actions because prior research has shown that continuous latent action spaces suffer from a problem known as “shortcuts” [Ye et al., 2023]. In this issue, latent actions only capture information corresponding to the immediate next step, ignoring broader contextual information and hindering the ability of latent actions to generalize well.“
>2.这篇文章所用的action模型经过了**大量预训练**。思考按NEST的构想，action能否从头训练成功收敛？
