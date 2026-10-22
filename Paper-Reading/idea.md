也可以叫：多轮智能体步骤-词元级优势的互补机制研究
https://chatgpt.com/s/t_6ad39588efe881918f8d7bd17bafc971
https://chatgpt.com/s/t_6ad39513f9448191a6d604fa305234f3
https://chatgpt.com/s/t_6ad38f07b6ec8191906a3196cd24cdb0
https://chatgpt.com/s/t_6ad3945e9d4881918d700be270e2d747
https://chatgpt.com/s/t_6ad393f6ccb08191ada7c8f322bb5b67

目前需要参考的几篇文章：
[VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents](https://arxiv.org/pdf/2610.08402) VETTA，turn-token分层设计，两个critic前半部分共享，取自于生成Transformer的前几个block，后面各自接独立小头，并且算是讨论了以什么方式融合效果更好
[Dependency-Aware Reward Shaping for Agentic Reinforcement Learning](https://arxiv.org/pdf/2610.01207)
[Stepwise Intrinsic Rewards for Reasoning in Large Language Models](https://arxiv.org/pdf/2602.01034)
如何处理一个turn中对CoT和action本身的信用分配关系？$A_{CoT}$由生成这部分推理后生成正确答案的概率的提升计算得出（简要原理，实际上机制更复杂）$A_{out}$则由结果的正确性和格式正确性，以类似GRPO的方式给出评分（见同路径下的MIG文件）
https://chatgpt.com/s/t_6ad843b5d5a08191bc0d786dead27ae1

1.看了一下符号学规划的相关论文，和细粒度奖励分配结合的点似乎并不多？看过的两篇都是PDDL和NL的转换或给出PDDL的策略验证可行性云云，并且都是带反馈的单步输出（不存在agent和环境的不断交互，除非提出新的交互方式说给环境输出之前先采用PDDL转换之类的）

2.但是之前的“研究问题”阐述确实是不足的，细粒度的需求本身不足以支撑我越过这么多已经存在的细粒度方法去硬把两个粒度结合起来，可以改为研究两个层级的分配存在什么关系，哪部分真的发挥了作用之类的
-可以通过访问alfworld的状态，得到每一个动作步前后增删的状态事件，如果哪个动作步后出现了属于任务成功状态的状态事件，可以判定它是否应该具有高优势

3.类似于GiGPO的做法，LLM的输出是包含CoT的，是否应该对这个CoT也进行奖励分配和决策归因？尤其是隐空间下，动作向量只代表动作本身还是包含整个CoT的整体？？
https://chatgpt.com/s/t_6ad8b05ca12c819192620324cf5e79de 
>[!NOTE] 注意上面这一点
>因为是开题报告里面没有体现的，是自己后面需要再思考一下去解决的问题




![[Pasted image 20261021121657.png]]
研究步骤级高优势的步骤是否“实至名归”可以看它对结果真正有用的状态是否增加。

但是词元呢，这一个词元的优势是否“应该”高只能看它是否是关键的谓词或者关键的对象，这是一个很主观的事情，并且只用有用状态增加（也就是整个步骤的结果）来说明词元应该优势高并不合理，因为这没有区分出步骤之中各个词元的区别。更进一步的，要想说明步骤和词元之间到底有甚么嵌套的联系，就更难站住脚了

--->参考MIG的想法，去计算每一个token出来之后得到当前动作的概率的变化，尤其是如果这个动作是导致了有用状态的增加的话，那么理论上说导致正确动作概率上升的token就应该是优势较大的。
![[Pasted image 20261022000455.png]]
鉴于agent过程中不存在一个每一步的标准动作答案：
![[Pasted image 20261022001106.png]]

这些基于PDDL状态的评估是一个反方向的，也就是并不是用PDDL状态来构建优势，而是在已有的优势构建方法下去反向地用这个方法去评估优势估计的合理性。