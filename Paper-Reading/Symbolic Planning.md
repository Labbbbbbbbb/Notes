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

