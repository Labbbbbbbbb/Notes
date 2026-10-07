Training Large Language Models to Reason in a Continuous Latent Space，Reasoning ai，post training（SFT，instead of RL）COLM2025

>[!NOTE]
>**We utilize the last hidden state of the LLM as a representation of the reasoning state (termed “continuous thought”). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. This latent reasoning paradigm leads to the emergence of an advanced reasoning pattern: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT.**


## related work
Latent reasoning in LLMs：
Shalev et al. (2024) discovered parallel latent reasoning paths in LLMs.发现 **LLM 在做隐式多跳推理时，不是串行锁定唯一中间结论，而是在隐层表征里同时并行地保留和处理多个候选中间结论（一个分布），再由后续层在这个分布上整体完成下一步推理。**
Goyal et al. (2023) pretrained the model by randomly inserting a learnable pause tokens to the training corpus. **pause token相当于在连续隐空间里给模型加了一段"静默思考时间"，不产生任何可见文本， 并且它是一个可学习的token，通过在训练中加入这个token，模型学会"往这些空槽位里放有用的中间计算"，是值得NEST借鉴的想法**

Wang et al. (2023) proposed to predict a planning token as a discrete latent variable before generating the next reasoning step.（我草！）

## Method
![[Pasted image 20261007153125.png]]
这个机制挺有意思的，传统LLM的递归输入模式相当于对前面的latent state一定要先用softmax先得出一个确定结果，再沿着这条前缀确定的路往下走，于此同时latent state中的其他可能全部被直接舍弃。但是第二种方法则是跳过softmax的输出过程，把信息丰富的隐藏状态直接往下传。

训练和推理的范式详见原文，这里作简要总结
训练是一个multi-staged过程，在第k个stage中规则把训练语料中的k个reasoning step的tokens换成k * c个latent thoughts（每个reasoning step是一段有逻辑的推理步骤，这个step的划分是在训练数据中就实现了的，k * c代表每个reasoning step分配c个latent thought来表达，c是超参），但是计算损失时不计算latent thought的损失，只计算tokens的交叉熵，latent靠可微的梯度回传。latent和token模式的转换由特殊token标记。
推理的过程，关于何时latent合适token有两种方式；首先开始的标记token bos是紧接在问题之后的，何时结束则有1，训练一个二分类器来决定和2，固定latent长度，写死。两种表现都挺好，所以为了简单用了后面一种

## Question
### 1.推理时在latent thought部分的输出只能作为下一个的embedding输入吧，而不能输出token，那么这个方法生成的序列是否比原始的CoT少一段呢，少的这一段无关紧要吗
![[Pasted image 20261007171439.png]]

![[Pasted image 20261007171512.png]]

个人感觉少的这一段不会特别影响观感，因为训练的时候也是把整个的reasoning step换成latent thought，所以我们能看到的只是显式的推理步骤少了几步，而不会出现一句话生成一半的情况
### 2.显然latent thought并非无结构意义的向量，它代表着当前时间步模型对输入token形成的最后一层隐藏状态，并且再过一个小小的输出头就是概率。那么直接用这个输出作为新的输入，是否对于模型本身是一个没法直接理解的意思？（一个向量，预训练模型本会把它理解为新token的embedding，但实际上它是杂糅了多种可能的thought）后训练阶段进行与预训练时意义不同为何不会带来混乱/崩溃？

**分布偏移（OOD）**：
- 预训练时，输入永远是 embedding 矩阵里的某个点（一条"词嵌入流形"上的点）；
- latent 模式喂进去的是经过 L 层非线性变换后的向量，通常**不在那条流形上**。
一个只见过"词嵌入"分布的模型，突然收到"off-manifold 的隐藏状态"，确实可能算出一堆没意义的东西。这个担心成立。
![[Pasted image 20261007172023.png]]
![[Pasted image 20261007172023.png]]

![[Pasted image 20261007172115.png]]

### 3.根据论文给出的结果，这样的输出的latent thought还能解码出真实token prob logit吗？为什么？？明明输入不完全一样了而且模型并没有针对latent的解码给出训练
震惊
![[Pasted image 20261007174254.png]]

![[Pasted image 20261007174430.png]]