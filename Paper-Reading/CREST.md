两个层级：
1.inter-turn
通过一个teacher模型（OPD）或者groun truth->self-teacher（OPSD）来提供不同trajectory之间的各个turn的价值高低（论文在related work中提到了之前的工作哪怕提供了turn级别的细粒度，也没有解决跨轨迹的turn的统一衡量的问题。）
2.intra-turn
针对OPSD在数学原理上容易为低熵的token算出更高的梯度的弊端，使用了token-level的熵掩膜

好聪明！
![[Pasted image 20260923151009.png]]