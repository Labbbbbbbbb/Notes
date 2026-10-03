两个层级：
1.inter-turn
通过一个teacher模型（OPD）或者groun truth->self-teacher（OPSD）来提供不同trajectory之间的各个turn的价值高低（论文在related work中提到了之前的工作哪怕提供了turn级别的细粒度，也没有解决跨轨迹的turn的统一衡量的问题。）
2.intra-turn
针对OPSD在数学原理上容易为低熵的token算出更高的梯度的弊端，使用了token-level的熵掩膜

好聪明！
![[Pasted image 20260923151009.png]]

CREST的机制是由环境的turn-level Reward得到优势的正负，然后利用 privileged self-teacher，并通过 entropy-gated modulation 调节这个 advantage 在不同 token 上的幅度（credit magnitude），最终得到 token-level credit。
所以粒度上：**session > turn > step > token**（**CREST / BFCL / WildToolBench 这类多轮会话场景**）：turn 和 step 严格区分如上（一个 turn 内含多步工具链）。。CREST 的两层信用分配正好对应：
- inter-turn：解决"哪个 turn 成功/失败"（环境只需在 turn 边界给 Rk​）；
- intra-turn：解决"这个 turn 内哪个 token 更关键"（self-teacher+entropy-gate 负责，环境不参与，并且计算magnitude的粒度是直接到token级别的）。
呃这个啥比说法，我一直以为turn就是step的。。。

magnitude的计算机制如下，重点在于只有当teacher的方向和verifier方向一致时梯度才会被放行，见下方第2、3点分析。
![[Pasted image 20261003211516.png]]