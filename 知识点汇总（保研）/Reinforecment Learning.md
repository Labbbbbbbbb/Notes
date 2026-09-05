# Algorithms
## DQN

![[Pasted image 20260603002827.png|611]]

> loss function是基于动作价值的贝尔曼最优方程和TD学习的，梯度下降更新
> 
> offline policy，行为策略是ε-greedy，这样既能探索，又能利用当前最优动作；目标策略是greedy 策略，即选择最大 Q 值动作，可以利用Replay-buffer学习。
> DQN的策略是选择最大Q值的动作，只能处理离散动作空间。（但是状态空间是连续的，不像Q-learning）
> DQN的目标策略更新是硬更新，n步进行一次，而不是后来那些（如DDPG、SAC等）采用的软更新（ema更新）


## PPO
关于PPO与vanilla policy gradient的区别：
policy-gradient:
$$
\hat{g}
=
\hat{\mathbb{E}}_t
\left[
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
\hat{A}_t
\right]
$$
$$
L^{PG}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\log \pi_\theta(a_t \mid s_t)
\hat{A}_t
\right]
$$
而PPO(暂时未clip版本，也就是CPI目标)与这个方向某种程度上完全一样：
$$
\hat{g}
=
\hat{\mathbb{E}}_t
\left[
\frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}  
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
\hat{A}_t
\right]
$$
$$
L^{CPI}(\theta)
=
\hat{\mathbb{E}}_t  
\left[  
\frac{\pi_\theta(a_t \mid s_t)}  
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}  
\hat{A}_t  
\right]
$$
区别只在于PG没有区分$\theta_{old}$与$\theta$ ，只要把PPO的$\theta_{old}$也变成$\theta$就会得到一模一样的公式。
这说明它们的更新方向和梯度原理本质一样，但是策略更新都伴随着一定程度的distribution shifting，只有在Trust Region之内才能保证更新不会因为剧烈的分布偏移而失效

同样因为分布偏移的问题，策略梯度算法只能是on-policy的算法，并且因为没有考虑trust region，容易崩溃，因此论文原文说”vanilla policy gradient methods have poor data effiency and robustness“，以及”While it is appealing to perform multiple steps of optimization on this loss $L^{PG}$ using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates“，而PPO则因为已经提前限制了策略更新跨度，所以可以在一定的epochs内反复利用一轮采样数据，成为了半在线半离线(严格说还是on-policy)的策略，并且相对更鲁棒

PPO:
$$
L^{CPI}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\hat{A}_t
\right]
=
\hat{\mathbb{E}}_t
\left[
r_t(\theta)\hat{A}_t
\right]
$$
$$
L^{CLIP}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}
\bigl(
r_t(\theta),
1-\epsilon,
1+\epsilon
\bigr)
\hat{A}_t
\right)
\right]
$$
![[Pasted image 20260603234345.png]]

PS:: 严格来说：

PPO 是一种 Policy Optimization / Policy Gradient 方法
而不是一种 Actor-Critic 方法。
其原文的重点在策略网络如何更新而并未过多提及价值网络，只是在sec.5后面附带提及了一些计算Advantage的方法，包括GAE、finite-horizon estimators与learned state-value function V(s)的结合等等
![[Pasted image 20260603235322.png]]

## SAC
--off-policy、robustness、data efficiency

>Although the soft Q-learning algorithm proposed by Haarnoja et al. (2017) has a value function and actor network, it is not a true actor-critic algorithm: the Q-function is estimating the optimal Q-function, and the actor does not directly affect the Q-function except through the data distribution.

虽然有的算法同时有Actor和Critic网络，但是它们的梯度并不耦合在一起，而是分开优化：
```
td_target = rewards + gamma * critic(next_states)

td_delta = td_target - critic(states)

actor_loss = -log_probs * td_delta.detach()
```
此时策略网络的目的不是定义 Critic，而是：

> 从当前估计的最优Q函数中采样动作。
> the actor network as an approximate sampler, rather than the actor

**SPI: soft policy iteration**
优化对象：
$$
V(s_t)
=
\mathbb{E}_{a_t\sim\pi}
\left[
Q(s_t,a_t)
-
\log \pi(a_t|s_t)
\right]
$$
V的损失函数：
$$
J_V(\psi) =
\mathbb{E}_{s_t \sim \mathcal{D}}
\left[
\frac{1}{2}
\Big(
V_\psi(s_t)-
\mathbb{E}_{a_t \sim \pi_\phi}
\big[ Q_\theta(s_t,a_t) - \log \pi_\phi(a_t|s_t) \big]
\Big)^2
\right]
$$
贝尔曼迭代过程：
$$
\mathcal{T}^{\pi}Q(s_t,a_t)
\triangleq
r(s_t,a_t)
+
\gamma
\mathbb{E}_{s_{t+1}\sim p}
\bigl[
V(s_{t+1})
\bigr]
$$

Q的损失函数：
$$
J_Q(\theta) =
\mathbb{E}_{(s_t,a_t)\sim \mathcal{D}}
\left[
\frac12
\Big( Q_\theta(s_t,a_t) - \hat Q(s_t,a_t) \Big)^2
\right]
$$$$
\hat Q(s_t,a_t) = r(s_t,a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} \big[ V_{\bar\psi}(s_{t+1}) \big]
$$其中$\bar\psi$表示ema更新的V网络

SAC的引理证明了在这种迭代之下Q能够收敛到$\pi$的价值上
SPI的策略是:
$$
\pi_{\mathrm{new}}
=
\arg\max_{\pi}
\mathbb{E}_{a\sim\pi}
\left[
Q^{\pi}(s,a)
-
\alpha \log \pi(a|s)
\right]
$$
而可证明要使$\pi_{new}$是使得V最大的策略，只需
$$
\pi_{\mathrm{new}}
=
\arg\min_{\pi' \in \Pi}
D_{\mathrm{KL}}
\left(
\pi'(\cdot \mid s_t)
\;\Big\|\;
\frac{
\exp\!\bigl(Q^{\pi_{\mathrm{old}}}(s_t,\cdot)\bigr)
}{
Z^{\pi_{\mathrm{old}}}(s_t)
}
\right)
$$
the policy parameters can be learned by directly minimizing the expected KL-divergence，其中exp表示指数，Z是所有exp(Q)的和
$$
Z^{\pi_{\mathrm{old}}}(s_t)
=
\sum_a
\exp\!\bigl(Q^{\pi_{\mathrm{old}}}(s_t,a)\bigr)
$$
动作的优化目标：
$$
J_{\pi}(\phi)
=
\mathbb{E}_{s_t \sim \mathcal D}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\phi}(\cdot \mid s_t)
\;\Big\|\;
\frac{\exp\!\left(Q_{\theta}(s_t,\cdot)\right)}
{Z_{\theta}(s_t)}
\right)
\right]
$$
而这里的动作$\pi$是从正态分布中采样$$
a_t
=
f_{\phi}(\epsilon_t; s_t)
$$并且因为正态分布后梯度不能传递，所以这里采用了重参数化处理，$a_t=\mu_\theta(s_t)+\sigma_\theta(s_t)*\epsilon_t$,使得梯度可以通过正态采样传递
展开得到（注意下式中Z因为与网络参数无关所以可以直接省略掉）
$$
J_{\pi}(\phi)
=
\mathbb{E}_{s_t \sim \mathcal D,\,
\epsilon_t \sim \mathcal N}
\Big[
\log \pi_{\phi}
\big(
f_{\phi}(\epsilon_t;s_t)
\mid s_t
\big)
-
Q_{\theta}
\big(
s_t,
f_{\phi}(\epsilon_t;s_t)
\big)
\Big]
$$
对$\phi$求梯度:
$$
\hat{\nabla}_{\phi} J_{\pi}(\phi)=
\nabla_{\phi}
\log \pi_{\phi}(a_t \mid s_t)+
\Big(
\nabla_{a_t}
\log \pi_{\phi}(a_t \mid s_t)-
\nabla_{a_t} Q_{\theta}(s_t,a_t)
\Big)
\nabla_{\phi} f_{\phi}(\epsilon_t;s_t)
$$
注意这里策略与价值的网络之间求梯度时没有sg环节，都是耦合的

**注意，我们说SAC是一个off-policy算法** ，是因为SAC的policy是在拟合一个由 Q 定义的 Boltzmann 分布，它的本质是接近Q-based的，因为Q才是最终决定策略方向的。SAC训练过程中Actor与Critic交替更新，新的policy 不断往新的 Boltzmann 分布靠近（SPI）。

SAC的最终目标：：（一方面V网络拟合的就是Q+H，另一方面策略网络的目的就是最大化这个值）
$$
\max \big[ Q(s,a) + \alpha \mathcal{H}(\pi) \big]
$$

## QMIX
![[Pasted image 20260605191650.png]]
**Centralized Critic & De-centralized Actor**
与之对比的另外几种思想：
**IQL(Independent Q-Learning)**:各自为政，各自优化Q function，但是不保证收敛，因为agents同事们的剧烈变化意味着对每个agent而言环境也在剧烈变化
**COMA**:共享一整个Critic，各自有actor，且这个Critic能看到所有的agent动作和全局state并以此为输入（注意：Q-function的输入是是全局状态S和n个agents的联合动作空间，维度很高），在agents数量多维度高的时候训练困难
**VDN(Value Decomposition Networks)**
同样共享一个$Q_{tot}$，但是这个$Q_{tot}$是由n个$Q_{i}$简单求和得到的，并且每个$Q_{i}$的输入只有观测$o_t$和自己的动作空间$u_t$，维度远小于联合动作空间。但是因为简单加和的关系，无法表达agents之间复杂的相互关系。
$$
Q_{\rm tot}(\tau, \mathbf{u}) = \sum_{i=1}^{n} Q_i(\tau^i, u^i; \theta^i)
$$
**QMIX**: value function that can be factored into a non-linear monotonic combination of the agents’ individual value functions in the fully observable setting，接近VDN，但是每个$Q_i$是用一个独立的w&b网络组合起来，保证weights非负，而不只是简单地加和
$$
\mathcal{L}(\theta) =
\sum_{i=1}^{b}
\Big[
y_i^{\text{tot}} - Q_{\text{tot}}(\tau, \mathbf{u}, s; \theta)
\Big]^2
$$
$$
y_i^{\mathrm{tot}}
=
r
+
\gamma
\max_{\mathbf{u}'}
Q_{\mathrm{tot}}
\left(
\tau',
\mathbf{u}',
s';
\theta^{-}
\right)
$$
其中$\theta^-$是EMA更新参数，$y^{tot}_i$表示target network。
QMIX没有强调是在离散的动作空间还是连续的，但同理的，离散的策略就是$Max(Q_{tot})$ ，连续的就像SAC一样用重参数采样，拿个神经网络降kl散度

## MAPPO
**系统研究了哪些 trick 对多智能体 PPO 真正有效**, 本质上是类似COMA的结构，用ppo当actor。（IPPO则是各自用自己的critic)
#### Value Normalization
优势计算：使用未归一化的V-real，用GAE计算得到V-target
![[Pasted image 20260607143620.png]]
得到V-target后进行归一化，再由网络预测，也就是critic的预测内容是归一化后的价值
![[Pasted image 20260607143826.png]]
>个人感觉这种线性的归一化，作用一定都是为了解决**不同来源/不同阶段的信号幅度不一致**问题，如flow matching目标的归一化是为了同一多种数据集中的动作标签的幅度差异，让它能从不同的数据集中学到一样的共性。那么对于这种方差均值的归一化，假设全程收集到的V-real方差均值都不变，则这个归一化理论上不会有什么作用（只是把数值整体平移放缩而已），但是即使 r 的分布完全稳定，V_target 也会漂移，因为策略变好了，同样的状态能拿到更多累积回报，V(s) 估计值整体抬升。所以归一化的动机根本上是训练动态导致的非平稳性，而不是数据分布本身的方差问题。并且这种归一化不会改变V的相对大小，所以不会影响学习的大方向。但实际上这种处理确实压缩了一些信息（比如V变好的程度），只是基于对训练平稳性的权衡而采用的trick，不能让它学到更多有用的内容
>遇到 reward 或 value 尺度不稳定的情况，判断要不要加归一化的标准就是这个：被压缩掉的信息对任务是否真的无关紧要。

#### Input Representation to Value Function
Value-function的几种输入内容的效果差异
**IND** ：只输入每个 agent 的 Critic 只用自己的局部观测 oi
**EP（Environment-Provided）**：用环境提供的全局状态向量
**AS（Agent-Specific）**：把 IND 和 EP **直接拼接**，每个 agent 的 Critic 同时看到：自己的局部观测 + 环境全局状态
**FP（Feature-Pruned）**：从 AS 出发，去掉重叠的特征，只保留互补信息，是论文推荐的方案
#### epoch数、minibatch数、batch大小
总结：epoch和minibatch数都不应太大，因为MARL中本身环境变化比较剧烈，epoch大了会导致策略变化过大，minibatch过多会因为每个batch太小而导致方差较大，不利于优化
而batch较大总体上对训练有益，但太大会降低data efficiency

# 关于RL的几个主要分类

## on policy Vs off policy

on：策略梯度的那些、PPO
off：DQN、DDPG、SAC
**判断是否off-policy的标准是** 看Actor是怎么更新的
普通的策略梯度：
$$
\nabla_\theta J =
 \mathbb{E}_{a \sim \pi_\theta}
 \left[
 \nabla_\theta \log \pi_\theta(a|s)\, Q^{\pi}(s,a)
 \right]
$$
要求a必须来自$\pi_\theta$的采样，否则梯度就不对了：
$$
\mathbb{E}_{a\sim\pi_{\mathrm{old}}}
\left[
\nabla \log \pi_{\mathrm{new}}(a|s)\, A
\right]
$$

根本原因是PG直接基于当前各个动作的概率分布做梯度，因而也要求采样的动作概率必须与当前的策略的概率分布一致

与之形成对比的是DDPG这种：
$$
\nabla_\theta J = \mathbb{E}_{s\sim\rho^\beta} 
\left[
\nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}
\cdot \nabla_\theta \mu_\theta(s)
\right]
$$

训练的是基于状态和价值函数应该输出什么样的动作，而不需要考虑这个标签动作是否来自当前的策略（呃实际上DDPG的replay buffer中的动作样本只用来训练critic了跟actor没啥关系）
那么同理的，SAC亦然，actor更新时只有state来自replay buffer，根本不依赖$a_{buffer}$ ![[Pasted image 20260607130136.png]]

## model-based Vs model-free

## 连续动作空间 Vs 离散动作空间
### 连续：
SAC、DDPG(直接输出动作本身，也不需要像SAC那样重参数化，也不是像离散网络那样输出某个动作的概率，但也因此是确定性策略)

### 离散：
Q-learning(状态空间也离散)、sarsa(TD) 、DQN、REINFORE(策略梯度)
（前面两种都属于是建表来记录Q值或记录其迭代过程的那种）
## value-based Vs policy-based
### 基于价值：
如DQN、Q-learning等等，球得一个价值函数，策略就是找价值最大的动作

### 基于策略：


$$
J(\theta)=\mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]
$$

$$
\begin{align}
\nabla_{\theta} J(\theta)
&\propto
\sum_{s\in S}\nu^{\pi_\theta}(s)
\sum_{a\in A} Q^{\pi_\theta}(s,a)
\nabla_{\theta}\pi_\theta(a|s) \\&=
\sum_{s\in S}\nu^{\pi_\theta}(s)
\sum_{a\in A}
\pi_\theta(a|s) Q^{\pi_\theta}(s,a)
\frac{\nabla_{\theta}\pi_\theta(a|s)}{\pi_\theta(a|s)}\\&=
\mathbb{E}_{\pi_\theta}
\!\left[Q^{\pi_\theta}(s,a)\nabla_{\theta}\log \pi_\theta(a|s)\right].
\end{align}
$$
这是策略梯度的通用公式，区别在于用什么去估计此处的Q值（其实不止是Q，只要期望等于total return，这是RL优化的根本目标），如Monte Carlo、GAE、TD形式，优势函数形式等等，都是为了平衡偏差与方差的方法。注意这里的 $\pi_\theta$ 既是行为策略也是目标策略，既是采样所用的策略也是最终要优化的策略（on-policy）因为这个梯度的期望是基于这个策略求得，如果策略不一样会导致价值函数不匹配

### Actor-Critic
既学习价值函数，又学习策略函数
价值函数更新梯度：
$$
\mathcal{L}(\omega)=
\frac{1}{2}
\left(r+\gamma V_{\omega}(s_{t+1})-V_{\omega}(s_t)\right)^2
$$
$$
\nabla_{\omega}\mathcal{L}(\omega) = -
\left(r+\gamma V_{\omega}(s_{t+1})-V_{\omega}(s_t)\right)
\nabla_{\omega}V_{\omega}(s_t)
$$
这是其中的一种形式，基于时序差分的迭代，其中减号前面部分是目标部分，属于s·g部分
策略函数更新的部分就是基于这里学到的价值函数，带入上面的策略梯度去更新策略网络


# 一些概念
### 占用度量（occupancy measure）
归一化的占用度量用于衡量在一个智能体决策与一个动态环境的交互过程中，采样到一个具体的状态动作对（state-action pair）的概率分布。
占用度量有一个很重要的性质：给定两个策略及其与一个动态环境交互得到的两个占用度量，那么当且仅当这两个占用度量相同时，这两个策略相同。也就是说，如果一个智能体的策略有所改变，那么它和环境交互得到的占用度量也会相应改变。
- 强化学习的策略在训练中会不断更新，其对应的数据分布（即占用度量）也会相应地改变。因此，强化学习的一大难点就在于，智能体看到的数据分布是随着智能体的学习而不断发生改变的。
- 由于奖励建立在状态动作对之上，一个策略对应的价值其实就是一个占用度量下对应的奖励的期望，因此寻找最优策略对应着寻找最优占用度量。
强化学习任务的最终优化目标是最大化智能体策略在和动态环境交互过程中的价值。根据1.5节的分析，策略的价值可以等价转换成奖励函数在策略的占用度量上的期望
![[Pasted image 20260806145348.png]]
状态访问分布：其中P是t时刻处于状态s的概率
![[Pasted image 20260806165325.png]]
其与占用度量的关系是
![[Pasted image 20260806165502.png]]
同一个MDP过程中，一个占用度量对应着唯一的策略，若占用度量相同，则两个策略是一致的
### 马尔可夫过程：（状态，状态转移函数）
下一时刻的状态只和上一时刻有关，与更早的历史无关
### 马尔可夫奖励过程
在马尔可夫过程的基础上加入了奖励函数R与折扣γ，描述了状态自身的变化过程与这个过程中获得的奖励，和对对应状态的价值评价
### 马尔可夫决策过程
在马尔可夫奖励过程的基础上加入智能体的动作A，A会参与进奖励与状态转移---r(s,a);P(s'|s,a)

### 最优策略
最优状态价值函数、最优动作价值函数、贝尔曼期望方程到贝尔曼最优方程

> 但是比如V(st)和V(st+1),他们本身就是对不同的状态的描述，最终收敛值不一样也很正常，有什么必然满足贝尔曼迭代的理由嘛
> 这个问题问到点子上了,你的直觉"V(s_t) 和 V(s_{t+1}) 收敛值不一样很正常"是完全对的——但这恰恰不是 Bellman 方程要求的东西,你可能把"函数值处处相等"和"函数满足某个方程"这两件事搞混了。我把这个误解拆开说清楚。

**Bellman 方程从来没说过 V(s_t)=V(s_{t+1})**

Bellman 方程是:

$$V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$$

这是一个**关于函数 V^π 的方程**,不是说"每个状态的值都相等"。它说的是:V^π 这个函数在 s 处的取值,等于"这一步 reward + γ 倍的 V^π 在 s' 处的取值"这个式子。s 和 s' 完全可以取值天差地别(比如 s 是"游戏刚开始",V^π(s) 可能很低;s' 是"即将胜利",V^π(s') 可能很高),Bellman 方程对此毫无异议,它只要求这两个不同的数值之间满足这个特定的代数关系,不要求它们相等。

**类比一下,你对"收敛"这个词理解错了对象**

你说"V(s_t) 和 V(s_{t+1}) 最终收敛值不一样也正常"——这句话本身是对的,但"收敛"这个词在这里不是指"V(s_t) 会收敛到某个值,V(s_{t+1}) 会收敛到另一个值"这种逐点各自收敛的意思。真正在收敛的是**整个迭代过程 V_0→V_1→V_2→...**,每一轮 V_k 都是一个完整的函数(对所有状态都有定义),收敛指的是这个函数序列作为整体,越来越逼近另一个固定的函数 V^π——用的是 sup-norm(所有状态上最大误差)来衡量距离:

$$|V_k - V^\pi|_\infty = \max_s |V_k(s) - V^\pi(s)|$$

这个距离随着 k 增大趋于 0,意味着对**每一个具体状态 s**(包括 s_t 也包括 s_{t+1}),V_k(s) 都在逼近各自对应的、可能完全不同的 V^π(s) 那个值。s_t 收敛到它自己的极限,s_{t+1} 收敛到它自己的极限,这两个极限值当然可以(而且通常确实)不一样,这跟 Bellman 方程完全不冲突。

**那"必然满足 Bellman 方程"这件事的理由到底是什么**

理由不是"因为不同状态的值应该相等"(压根没有这个说法),而是"V^π(s) 这个数,按定义就是从 s 出发未来累积回报的期望",这个定义本身在数学上可以严格展开:

$$V^\pi(s_t) = \mathbb{E}\left[\sum_{k=0}^\infty \gamma^k r_{t+k}\right] = \mathbb{E}\left[r_t + \gamma\sum_{k=0}^\infty \gamma^k r_{t+1+k}\right] = \mathbb{E}[r_t] + \gamma,\mathbb{E}\left[\sum_{k=0}^\infty \gamma^k r_{t+1+k}\right]$$

后面这一项 $E[Σγ^k r_{t+1+k}]$,如果你从 s_{t+1} 这个状态往后看,它的定义恰好就是 $V^π(s_{t+1})$(只是外面还要对 $s_{t+1}$ 具体是哪个状态取期望,因为从$s_t$ 转移到 $s_{t+1}$ 是随机的)。所以这个等式纯粹是把一个无穷求和拆成"第一项+剩余部分",再指出"剩余部分"这个式子跟 V^π 的定义长得一模一样——这是一个**纯代数拆分**,不涉及任何"两个状态的值应该接近或相等"的假设,$V^π(s_t)$ 和 $V^π(s_{t+1})$ 数值上爱差多远差多远都行,方程只约束它们之间要满足"差 γ 倍再加一个 reward"这个关系,不约束它们本身的大小。

**一句话消解你的疑问**

你担心的是"必然相等"这件事,但 Bellman 方程要求的从来不是"相等",而是"满足这个递推关系式"——这个关系式对任意两个可以取任意不同数值的状态都成立,因为它就是从"未来回报之和"这个定义直接拆分出来的代数恒等式,不是外加的强约束,更不是要求不同状态的价值趋同。
### 动态规划算法：
需要已经知道状态转移函数和奖励函数，通过贝尔曼方程把当前问题和子问题拆开（比如当前状态的价值和下一个状态的价值），且子问题与当前问题之间存在递推关系，通过不断的迭代最终得到准确解它与后续算法的不同之处在于它对环境是全局已知的，所以不需要探索、不需要估计，可以得到唯一的最优准确解，也因此是一种“规划”算法
#### 策略迭代算法：
分为策略评估和策略提升两个阶段，策略评估阶段通过贝尔曼期望方程不断迭代，目的是得出准确的V函数；策略提升阶段将新得到的V用于采取策略（贪心算法，argmax_a r+γV)
#### 价值迭代算法：
只维护一个价值函数，不存在显示的策略。（策略就是采取最大价值的动作）可以看作是只做了一次策略评估的策略迭代算法
使用贝尔曼最优方程更新V
![[Pasted image 20260806190932.png]]
![[Pasted image 20260806191006.png]]

### 时序差分算法
比如Sarsa和Q-learning
![[Pasted image 20260806194412.png]]
α后面那一串也称为时序差分误差（TD error) 
注意：TD和动态规划最大的差别就是TD error并不是真正的价值期望，而只是对某一小部分数据的采样所得。
不同于策略迭代中对V更新很多次才进行策略提升，TD是一种价值迭代的算法，这是广义策略迭代的方法，即策略提升不需要等策略迭代完全做完才能进行，因此包括sarsa、包括后面见到的许多Q-base算法，都是更新一下策略、用新策略采下数据、再更新下策略。
此外，Sarsa与Q-learning都没有使用神经网络，而是由一个初始值一点点迭代出来的Q-table，只能用于离散的动作空间+离散的状态空间

### DQN
DQN是使用了深度神经网络的Q-base方法，它可以接收离散的状态空间，但是通常动作空间是连续的，因为基于Q的最优动作采样时$max_a Q$ 
![[Pasted image 20260807120224.png]]
### Double DQN
在DQN的基础上，把评估价值（计算target）的网络（价值网络）和用于采样动作（argmax_a）的网络（训练网络）分为两个网络
![[Pasted image 20260807121126.png]]
![[Pasted image 20260807122217.png]]
并且这里Qnet输出维度是action-dim，代表动作空间中n个动作各自的q-value。
action = self.q_net(state).argmax().item()

### Deuling DQN
把Q拆分为V+A的网络去训练，前面几层网络共享，后面的头分开。 Dueling DQN 能够很好地学习到不同动作的差异性，在动作空间较大的环境下非常有效。
"传统 DQN 里,除了共享层带来的被动漂移外,只有被采样到的动作的 Q 值获得了直接的、针对性的梯度校正"
这正是 Dueling DQN 想解决的问题。把 V(s) 单独拆出来一个分支后,不管这一步采样到哪个动作,V(s) 都会因为这条轨迹的 TD 误差被显式地、有方向地更新,而不是靠共享层间接扩散过去。这在动作数目多、且多数动作对状态价值影响差不多的场景下,能让价值估计的学习效率明显提高。

### 策略梯度
![[Pasted image 20260807135944.png]]
其中pi代表概率或概率密度
此时的概率（密度）梯度使得pi向倾向于采样Q值较大的动作的方向改变，使得策略更像是一个针对Q函数的采样器
这个形式的关键价值在于,它把一个"需要知道环境模型才能求"的梯度,转化成了一个**期望的形式**,而这个期望里的采样分布 s~ν^{π_θ}, a~π_θ(·|s),恰好就是你让 agent 实际跟环境交互时自然产生的数据分布——你不需要显式知道 ν 或 P 长什么样,只要让 agent 按当前策略去跑,跑出来的轨迹里状态出现的频率,自动就近似了 ν^{π_θ}(s),动作出现的频率自动就是 π_θ(a|s)。于是整个梯度可以直接用蒙特卡洛采样去近似
```
G = 0

        self.optimizer.zero_grad()

        for i in reversed(range(len(reward_list))):  # 从最后一步算起

            reward = reward_list[i]

            state = torch.tensor([state_list[i]],

                                 dtype=torch.float).to(self.device)

            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            log_prob = torch.log(self.policy_net(state).gather(1, action))

            G = self.gamma * G + reward

            loss = -log_prob * G  # 每一步的损失函数

            loss.backward()  # 反向传播计算梯度

        self.optimizer.step()  # 梯度下降
```
` log_prob = torch.log(self.policy_net(state).gather(1, action))`获得已经发生的动作的概率值，计算更新梯度（因此要求是on-policy）
log_prob就是由当前策略算出的概率，policy_net(state).gather(1, action)得出action_list中已有的动作，用这些动作得到计算梯度。
同时策略梯度直接提升的是缓存区中存在的动作a（这个a的Q值大就提升它的概率，提升的同时自然也会压低别的动作的概率）
同时注意 G = self.gamma * G + reward 这里采用的是蒙特卡洛的估计法，是从reward list倒着走完全过程获得了真实的完整轨迹之后才在最后一起optimizer.step的，而不像TD算法/价值迭代可以使用尚未收敛的估计值V(s)来进行自举更新

**AC之所以是在线策略：对于一批数据，它需要用策略网络去计算其中动作的概率，从而计算梯度，然后因为原理上这个梯度应该是A* log的表达式对pi的期望，所以只有当这些动作数据来源于pi本身时，这个梯度的方向才是对的 
而且不同于Q-base的方法，策略梯度不能通过现场采样动作来避免对历史动作数据的需求，因为相应的状态和奖励数据都是与当时的动作数据息息相关的**

### Actor-Critic
![[Pasted image 20260807154219.png]]
其中计算actor loss时时序差分残差做了detach处理
- Actor 提供 on-policy 的数据分布,让 Critic 能估计出"在当前这个策略下"的 V/Q(这点很关键,Critic 估计的是 V^{π_θ},策略一变,V 的真值也变,所以 Critic 必须靠 Actor 不断提供新策略下的新数据才能保持准确)
- Critic 反过来把这些数据提炼成一个低方差的评价信号,加速 Actor 的更新

### PPO-Clip 近端策略优化
![[Pasted image 20260807155323.png]]
min(CPI_Target,clipped_target)
这是优化目标，倘若A为正而比值r超过1＋ε，则引起clipped，但反之，若此时它小于1-ε，就会因min的作用而不被clip

连续动作情况下的PPO：actor网络输出变成μ和σ，action是基于它们的正态分布（动作空间连续）
```
action_dists = torch.distributions.Normal(mu.detach(), std.detach())

old_log_probs = action_dists.log_prob(actions)
```
action_dists 是一个概率分布对象(比如 torch.distributions.Categorical(probs) 或者连续动作用的 torch.distributions.Normal(mu, sigma)),不是一个具体数值,而是一整个分布,支持采样、算概率密度这些操作。

actions 是这一批数据里实际执行过的动作(从 replay buffer 或者这一轮 rollout 里取出来的,是历史数据,不是现在重新采样的)。

.log_prob(actions) 就是把这批 actions 代入这个分布,算出每个动作在这个分布下的对数概率 log π_θ(a|s)。
**注意PPO连续动作空间不需要像SAC那样采用重参数化技巧，它虽然是输出了正态分布的均值方差，但是梯度计算并不是用这个正态分布采样的动作（因此不像SAC那样面临一个导数的截断问题），而是用均值方差的到一个概率分布函数，计算已经存在的动作的概率密度，这一步是可导的**
### DDPG 深度确定性策略梯度
![[Pasted image 20260807164332.png]]
这是一个对Q函数得耦合链式求导（对Q中得a求偏导，其中a等于μ，是策略函数）
![[Pasted image 20260807170605.png]]
Q网络参数有另外得更新，其中回放池的at就是用来更新Q的，它只判断某个s-a对的价值，不在乎这个a是从哪里来的，在策略网络算梯度时，所使用的a来自策略网络现场采。总的来说，DDPG更像一个带独立策略网络的Q-base策略
策略与价值深度耦合，更新时不带detach
为增加随机性，输出动作会添加噪声
### Soft Actor-Critic ->最大熵强化学习
![[Pasted image 20260807172930.png]]
α代表探索程度
Q的损失函数：
（上文那个版本有独立的V网络，但是在后续的进展中就直接用Q+H代替了）
注意：这里的逻辑是先定义了最终目标V=reward+H，由此动作价值的定义等于r加未来策略期望下的状态价值V，由此有了Q=r+γV的迭代方程
![[Pasted image 20260807193139.png]]
策略损失函数：
![[Pasted image 20260807195440.png]]
策略pi最终拟合了Q的玻尔兹曼分布（一方面这个损失函数梯度会让SAC的终极目标上升，另一方面min$L_\pi$是一个带约束的泛函极值问题，用拉格朗日乘子法,对pi求变分导数，可求得其满足玻尔兹曼分布）
**并非所有策略最终都应该满足符合Q的玻尔兹曼分布，它只是最大熵目标V下的特定形式**

SAC的连续动作空间和PPO一样都是输出均值方差用正态分布采样，但是计算梯度的原理不同，SAC的熵项的求导需要直接对$\pi_\theta(f_\theta)$求导，其中耦合的$f_\theta$是策略输出的动作，这一部分若没有重参数化采样，是不可导的。**也就是说不可导的是$f_\theta$，而不是$\pi$这个概率本身**
注意到SAC的策略损失函数中也不需要用到历史缓冲区中的action，都是现场采样的
### GAE广义优势估计
优势的定义是Q-V，但有时没有单独训练Q、V网络或者足够的数据量计算期望
经常也用时序差分误差来替代优势的位置
```
advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
```


### offline RL
#### BCQ，批约束下的Q-learning
策略由两个部分组成：主要网络$G_w$ 和微调网络 $\epsilon_\theta$ 
主要网络$G_w$是一个条件生成模型conditional VAE，用于以模仿学习的形式先学习以state为条件的action分布。这一部分完全基于对(s,a)专家动作的学习，与Q和R无关。
微调网络$\epsilon_\theta$则是基于Q-learning调节的，随着Q的拟合，$\epsilon_\theta$ 的优化方向就是让动作往Q高的方向走。最后的策略动作是把$\epsilon_\theta$ 与$G_w$相加。
## 读论文.2026.8.8
### Flow-GRPO
任务：text-to-image (T2I) generation
challenges: (1) Flow models rely on a deterministic generative process based on ODEs , meaning they cannot sample stochastically during inference. In contrast, RL relies on stochastic sampling to explore the environment, learning by trying different actions and improving based on rewards.This need for stochasticity in RL conflicts with the deterministic nature of flow matching models.
-->**ODE-to-SDE strategy**, converting the ODE-based flow into an equivalent Stochastic Differential Equation (SDE) framework
(2) Online RL depends on efficient sampling to collect training data, but flow models typically require many iterative steps to generate each sample, limiting efficiency.
**Denoising Reduction strategy**, "We find that online RL for flow matching models does not require the standard long timesteps for training sample collection."
::We show that the Kullback-Leibler (KL) constraint effectively prevents reward hacking, where reward increases at the cost of image quality or diversity.

GRPO: GRPO的设计与PPO非常像，都是策略梯度和clipped，区别在于advantage是如何计算的。PPO通过训练critic网络计算优势，但是对于动作空间和状态空间较大的场景，训练critic非常昂贵，而GRPO是一个更轻量的，他的优势来源于组内的各个动作的reward比较，因此不需要独立的critic ，但是注意GRPO也是在线用法
“Unlike other policy based methods like PPO , GRPO   provides a lightweight alternative, which introduces a group relative formulation to estimate the advantage.”

##### Flow-GRPO优化目标：
累计奖励，训练策略与参考策略的KL散度（参考策略指RL优化前的网络）
![[Pasted image 20260808154913.png]]

这篇文章把 Flow Matching / Diffusion 模型的迭代去噪过程重新解释成一个 MDP（Markov Decision Process），从而可以用强化学习的方法优化生成过程。
过程：以当前噪声$x_t$、条件c、当前时间步t为状态state，策略生成的动作是下一步降噪后的样本$x_{t-1}$ ，到最后一步$t_0$结束后会获得奖励r
（reward只在最后一步产生，但计算出的$trajectoryadvantageA^i$会广播到所有denoisingtimestep。​）
![[Pasted image 20260808160338.png]]
也就是说：policy就是 diffusion model。**并且因为flow matching基于ODE，故而状态转移函数也完全确定**
这一条RL的episode其实就是diffusion trajectory
由于GRPO是在扩散最后一步才会针对结果给出reward从而计算advantage，所以相对会比较稀疏，但由于FlowMatching的ODE步数相比起传统RL甚至diffusion的步数都要小挺多，所以（也没那么稀疏）
是稀疏奖励，但由于 diffusion trajectory 短、模型已有先验、GRPO共享trajectory advantage，所以仍然可训练
如何更精确地进行 denoising step 的 credit assignment，是 diffusion RL 当前的重要研究问题

##### From ODE to SDE.
$$dx_t = v_tdt$$由确定性速度场变成-->
![[Pasted image 20260808164822.png]]
并且仍满足"matches the original model’s marginal probability density function at all timesteps"
也就是说这个score项并不是flowmatching学习出来的，是从ODE变为等价的SDE为了保证想通过的边缘分布所做的补偿（两个过程的$p_t(x)$相同）
上式score前面是减号是因为t是从1到0的
![[Pasted image 20260808181228.png]]
为何要保持边缘分布不变？
RL 本身原理上并不要求中间状态分布和原模型一致。但是对这种两阶段 RL 来说，最大的好处是保留 pretrained policy 的行为分布，让 RL 优化始终在 pretrained flow 熟悉的状态空间里面进行，由此，使用和原来策略的KL散度来约束新策略才有意义，也才能减小分布漂移所带来的reward hacking
>but....?
> **保持边缘分布 pt​(xt​) 并不等价于保持原来的路径语义（path semantics）。** pt​(xt​)只是**单个时间切片的分布**，它确实把路径信息积分掉了。所以这里的保持边缘分布的意义是》》



**score函数的物理意义**
定义：$s(x)=∇_x​logp(x)$ 描述：概率密度在当前位置增加最快的方向​

举一个一维的例子：假设 $p(x) = \mathcal{N}(0, 1)$
那么：$log p(x) = -\frac{x^2}{2} + C$
求梯度：$\nabla \log p(x) = -x$
如果：$x = 3$ 那么：$\text{score} = -3$
意思是当前位置在右侧：梯度告诉你往左走。
如果：$x = -3$, $\text{score} = 3$告诉你往右。
所以score 是一个概率恢复力（force）
score学习的是概率密度的局部梯度场，而不是直接学习数据点位置。​
反向生成时，利用这个梯度场，把随机噪声逐步引导回高概率的数据流形附近。

### DPO 
![[Pasted image 20260808193214.png]]
一种在offline数据上的微调，类似SFT。直接通过降低不喜欢的数据的生成概率，提高偏好的数据的概率来进行微调，也不需要再与环境交互。
### DreamerAD
自动驾驶存在长尾问题：存在一些极端少见情况却会引发严重后果，因此需要保证模型在极端情况下也能稳定。
但是一般的深度学习往往很少关注长尾问题，因为如果训练数据少就难以习得这方面的特征。而强化学习则更擅长解决这方面的问题。

提出问题：1.真实世界的自驾RL成本高风险大，传统的仿真则会引入sim2real的偏差
2.基于视频的世界模型可以用来做imagination-based的策略学习，但是(1)视频用diffusion生成，采样速度很慢；(2)视频生成模型很多时候更关心生成的视觉保真度、美观性而不是对空间和运动的理解能力。

世界模型SF-WM：
以Epona为基础，可以理解为一个 **Action-conditioned Driving World Model（动作条件世界模型）**。an autoregressive diffusion model based on flow matching
同时把环境动力学模型（world model）和策略模型（policy/planner）放进同一个可学习框架中，让二者共享对未来世界的理解。unifies video generation and trajectory planning, enabling future video prediction conditioned on action controls

**输入：过去的帧和动作；输出：未来的帧和动作**

这个模型中的shortcut是指step distillation.具体实现是：生成模型的输入中可以选择跳跃的间隔，若跳跃间隔小于dmin，则使用原始的flow matching进行更新，若大于dmin，则使用教师模型生成两个半步的速度的平均值来当拟合的v目标
![[Pasted image 20260808223959.png]]
>说是two teacher half-steps，但是其实上面teacher用的网络不是和下面训练的网络一摸一样吗??所以是用这种自一致性的方式**强迫模型 $\phi_\theta​$ 学会“一步跨大步”的预测能力，等同于“分两步跨小步”的精细结果？**

奖励模型ADRM
