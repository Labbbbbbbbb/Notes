## softmax的temperature处理
![[Pasted image 20260930220843.png]]
OPD论文中是这样处理温度$\gamma$的，训练时temperature置为1，$\gamma$=1 就是 softmax 的原始定义——“模型实际输出的概率分布”。训练时 student 保持$\gamma$=1 的真正原因是：$\gamma$ 不是模型参数，它只是部署/采样时套在 logits 外面的一个后处理旋钮。如果训练时让 student 在 γ≠1 下优化，那么权重学到的 logits 会去“补偿”这个 γ → 训练好的模型只有在该 γ 下才给出正确分布，于是部署时如果用 γ=1（或 greedy），分布就错了。模型权重应该学习“真实分布”（γ=1），温度只在需要随机性的环节临时施加，这样训练完的模型在任意推理温度下都是自洽的。这是技术约束，不是风格选择。

同时要区分training中的温度有rollout和loss计算两个部分
（注意rollout是由于on-policy的设计，训练数据是由student-policy自己生成的）
![[Pasted image 20260930221237.png]]

**Teacher 的温度是另一回事（而且≠1）**
经典 Hinton (2015) 蒸馏里，teacher 常用 **T>1** 把分布软化，让暗类别（dark knowledge）的相对概率暴露得更清楚。
GKD 这篇的实验实际做法是 **teacher temperature = 0.1**——反方向，把 teacher 分布压得更尖锐、反馈更确定（因为 on-policy 场景下 student 的前缀往往是 teacher 没见过的，尖锐反馈能给明确纠正信号）。所以：
```text
Student γ = 1     ← 不动，保证学的是标准分布
Teacher γ = 0.1   ← 可调旋钮，按任务需要控制反馈软硬
```
**温度调节的自由度全给了 teacher 侧，student 侧锁死**——这是蒸馏文献里的通行设计模式。

**Evaluation时的两种选择：这部分确实是 benchmark 惯例**
 Greedy（γ→0）时数学上是个极限：γ→0⇒$p_γ$​→$one-hot(argmax z_i​)$
整个 softmax 坍缩成 argmax，输出**确定性、可复现**。论文报告 XSum/GSM8K 用 greedy、WMT 用 beam search，原因是：
- Benchmark 分数需要**可复现的点估计**，否则每次采样分数都波动，方法之间无法公平比较
- Greedy 代表模型的“单次最佳发挥”
**Temperature sampling（γ>0**
用于考察**分布层面的性质**：多次采样下的质量、多样性、是否真的逼近 teacher 分布。GKD 的优化目标本来就是分布匹配（divergence），只用 greedy 评估只能看到模态顶端，所以需要 γ>0 的采样评估作为补充。
