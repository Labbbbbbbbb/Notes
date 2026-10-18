Paper1. Towards Reliable and Robust LLM Planning:  A Symbolic Feedback-Driven Iterative  Self-Refinement Framework
Paper2. 


# Paper1
## 什么是Planning（规划问题）
规划问题可以描述为一个三元组问题$P = (\Xi, \mathcal{I}, G_{trn})$
![[规划问题思维导图.png]]

ps.这里有一个有些容易抽象的概念就是状态s，状态是当前为真的命题变量集合，而不是其中的一个变量，比如状态空间是 $\mathcal{V} = \{at(apple,table), at(apple,shelf), handempty, holding(apple)\}$，那么具体的状态可以是$s=\{at(apple,table),handempty\}$

>[!NOTE] Planning的本质是在：
>A planner Π solves a confronted planning problem P by crafting an appropriate plan π = Π(P ) usually represented as a sequence of actions π = (a0, a1, . . . , an), which transforms the initial state I into the goal state Gtru by sequentially applying those actions in π.

## 什么是PDDL
- PDDL为上述问题定义提供具体语法，用来表示其中的 Domain、初始状态、目标和动作规则。
- Planner（规划器）读取 PDDL 描述的规划问题，通过搜索或推理找到满足目标的动作序列。
![[Pasted image 20261017150344.png]]
![[Pasted image 20261017150631.png]]

**The structure of PDDL is generally divided into two components: domain definition and problem definition. The domain definition captures the shared characteristics of a class of planning problems, including type specifications, predicate definitions, and action schemas. In contrast, the problem definition specifies a concrete planning instance, including the initial state, goal conditions, and the set of involved objects.**

例如：对于一个场景domain和problem的定义分别如下
```
# domain中的action,相当于只有格式但是不给具体的物品/位置
(:action pick-up
  :parameters (?x - item ?l - location)
  :precondition (and (at ?x ?l) (handempty))
  :effect (and
    (holding ?x)
    (not (at ?x ?l))
    (not (handempty))
  )
)   #此外还应包含put-down，此处没写
```
即：$$
\begin{aligned}
name_a &= \text{pick-up} \\
pre_a &= \{at(x,l), handempty\} \\
eff_a^{+} &= \{holding(x)\} \\
eff_a^{-} &= \{at(x,l), handempty\}
\end{aligned}
$$
```
# problem
(:objects apple - item
          table shelf - location)

(:init
  (at apple table)
  (handempty)
)

(:goal
  (at apple shelf)
)
```
即：
$$
\begin{aligned}
\mathcal{I} &= \{at(apple,table), handempty\} \\
G_{tru} &= \{at(apple,shelf)\}
\end{aligned}
$$
规划器应该求出的动作：
$$
\pi = (pick\text{-}up(apple,table), put\text{-}down(apple,shelf))
$$


## Methodology
### Framework
![[Pasted image 20261017225653.png]]

![[Pasted image 20261017230852.png]]
### Natural Language Describption
把pddl翻译为自然语言，输入LLM

### Feedback生成过程
![[Pasted image 20261017225732.png]]
Syntax：作为PDDL的语法是否正确
VAL：执行的动作是否存在先决条件不符合等问题
Recogizer：根据现有条件判断最可能的目标是不是PDDL中的真正目标

>[!NOTE] 
>但是值得注意的是这篇文章里面同样没有训练的过程，对于LLM的工作属于prompt engineering，相当于用一个反馈机制不断地得到对prompt的构造。并且除了LLM存在神经网络，其他的流程基本是基于规则的构造和映射。(并且因为我对这个规则并不了解，我感觉这个映射过程有点神奇)



## Question

### LLM的直接输出应该是NL吧？用什么机制或规则把它重新映射为PDDL呢

### 最终的输出策略也是PDDL形式吗？还是NL但也和ALFWorld一样有严格限制？如果是PDDL那是由LLM生成后转换的还是让LLM直接生成？

### 有些问题放在论文的一些个批注里

### 所以这篇论文其实有点agent的World Model的味道，planner一次性给出整条策略序列，并通过符号机制（相当于一个先验、对未来可能的建模）来反复修改策略序列并将其输出
对，你这个理解基本正确，而且已经抓住了这篇论文与 World Model-based Planning（基于世界模型的规划） 之间的联系。
不过需要做两个区分：
1. PDDL 符号系统确实可以被视为一种显式的、基于规则的世界模型。
2. 这篇论文更准确的定位仍然是 Model-based Planning + Symbolic Verification + Iterative Refinement，而不是通常所说的学习式 World Model。

https://chatgpt.com/s/t_6ad3945e9d4881918d700be270e2d747


# Paper2
Toward Secure and Reliable PDDL Formalization  of Large Language Models with  Planner-in-the-Loop Feedback

Benchmark构建+基于这个Benchmark的新训练框架、
这篇论文的核心任务是让 LLM 更好地把自然语言（NL）转换成 PDDL，而不是训练 Planner 生成更好的计划。 Planner 是外部的符号工具，负责验证 LLM 生成的 PDDL，并提供可解性和计划质量方面的反馈。
>[!NOTE] 注意
>**这篇论文的核心任务是让 LLM 更好地把自然语言（NL）转换成 PDDL，而不是训练 Planner 生成更好的计划。 Planner 是外部的符号工具，负责验证 LLM 生成的 PDDL，并提供可解性和计划质量方面的反馈。**
## Method
![[Pasted image 20261018153015.png]]

### 训练推理时
![[Pasted image 20261018133636.png]]

（On the learning side, we combine supervised fine-tuning with planner-grounded preference optimization）（使用下面bench自己构造的数据）
- SFT：用 NL–PDDL 配对数据训练 LLM，让它学会从自然语言生成 PDDL。
- DPO：通过规划器验证结果构造偏好数据，让模型倾向于生成可求解、规划质量更好的 PDDL。
（at inference time, if parsing or planning fails, the system converts diagnostic logs into structured feedback and prompts the model to perform localized repairs until verification succeeds or a fixed budget is reached.）
- 推理时修正：模型部署或评测时，再通过实际的 Planner-in-the-loop 反馈迭代修改错误输出。
因此，论文的整体思路可以概括为：
构建 NL–PDDL 数据 → 训练 LLM 做 NL-to-PDDL 转换 → 用规划器结果提供训练信号 → 推理时通过规划器反馈修正输出。

### 数据构建时（Benchmark）
![[Pasted image 20261018133734.png]]

![[Pasted image 20261018144312.png]]

DPO阶段数据构建：(离线DPO)
![[Pasted image 20261018144430.png]]
![[Pasted image 20261018152211.png]]
参考 PDDL $y^+$ 能让规划器找到有效计划，例如先装货、再驾驶、最后卸货。
然后，程序化扰动可能生成一个候选 $y^-$，例如删除 `(Connected A B)` 这一初始事实。
如果领域动作规则要求道路连通才能驾驶，那么这个候选就可能无法完成任务。经过验证后，它便可以作为负样本。
DPO 学习的是：$y^+ \succ y^- \mid x$
也就是：对于同一个自然语言任务 \(x\)，模型应该更偏好正确的参考形式化，而不是这个较差的候选。
这里需要注意两个细节：
- 负样本不一定全部是语法错误。一个 PDDL 完全可以解析，但仍然不可解。 
- 并不是所有扰动都自动成为负样本。论文描述的是经过 Validator 和 Planner 筛选，保留不可解或规划质量明显差于参考结果的候选。
所以，Planner 的价值在于提供可执行的筛选依据，而不只是让模型自己判断哪个答案更合理。

Benchmark打分工作：
对每个实例，记录自然语言输入 $x_i$、参考 PDDL $y_i$ 和模型输出 $\hat y_i$。
然后计算四个指标：
![[Pasted image 20261018152412.png]]

## Question
### 1.我有点没懂，我觉得这个构建数据的构成和评估的过程其实很割裂，我觉得只能说是一个方案但是称不上是一个合体的benchmark，我感觉这个打分的东西和生成数据的完全是可以拆开的两个部分。是我哪里不了解没get到吗？并且这个打分其实就是最后给出一个数据吧，这个数据也并没有再一次作为反馈去给到训练环节。
![[Pasted image 20261018155907.png]]
![[Pasted image 20261018161814.png]]

>[!NOTE]
>反馈阶段也是调用 NL-PDDL-Bench 所采用的同一套符号工具链，但 iterative repair 本身属于论文的推理方法，不是 Benchmark 数据集自带的功能。但是这一套统一的工具链也是构建这一套基础设施（bench，数据集，反馈，评分）的意义所在。
### 2.等一下？所以这篇论文的任务是让Planner生成更好的计划还是让LLM能更好地把NL转换成PDDL？它注重的是哪方面的训练？
这篇论文的核心任务是让 LLM 更好地把自然语言（NL）转换成 PDDL，而不是训练 Planner 生成更好的计划。 Planner 是外部的符号工具，负责验证 LLM 生成的 PDDL，并提供可解性和计划质量方面的反馈。![[Pasted image 20261018160620.png]]

### 3.文中提到了multi-domain，后面的实验也又cross-domain，所以训练的时候是一起在多个domain上一起训练然后同时发现它在没见过的domain上也可以表现得比较好吗
https://chatgpt.com/s/t_6ad487009ad48191a5e1bb9bb5e875cf

