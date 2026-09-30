**"The key idea is that the baseline choice controls how many samples receive positive vs. negative advantages, which directly impacts exploration behavior. Specifically, a lower K marks more samples as having positive advantage, encouraging the model to exploit these successful patterns and reducing entropy.Conversely, a higher K makes fewer samples appear successful, pushing the model to diversify its behavior patterns, thereby increasing entropy.By tuning the quantile parameter K, we can control the exploration-exploitation balance"**

**"This mechanism has a striking empirical consequence: it naturally sparsifies updates. With a tuned K, roughly 80% of responses receive zero advantage."**

**“Overall, QAE reframes entropy regulation as a baseline-design problem rather than a token-level tuning problem.”**

**“The explosion is mechanically rooted in the advantage baseline, which systematically mishandles negative-advantage samples under reward outliers.
The issue is therefore a baseline-design flaw, not a hyperparameter tuning problem at the token level.”**

？![[Pasted image 20260928153424.png]]

![[Pasted image 20260929144625.png]]
这里的Ri是二值化的0/1，当回答中得分为0的个数多于K%时，K百分位数也就是baseline为0，此时为Hard模式，鼓励positive advantage的答案，强化正确轨迹，熵倾向于下降（只推尖成功的分布，不压散错误的分布，防止exploision）；
随着训练进行得分为1的比例变多，baseline变为1时原来的正确答案优势就为0，错误答案优势为负，此时熵倾向于上升（已经成功的不推，把错误压散，此时熵上升，避免坍缩）
![[Pasted image 20260929233327.png]]

这个机制挺有意思的，它利用这个二值化的特点，把原有的mean-baseline改成K-percentile，baseline的自然变化恰好把训练分成了上述两个部分，只有初期Hard模式的成功回答和后期Easy模式的失败回答会获得非零梯度，且前期推尖防止熵爆炸保持稳定，后期压平防止熵坍缩保持探索，恰好各司其职实现了双端的保护。
针对这一点文中给出了关于熵的理论证明
![[Pasted image 20260930000012.png]]

后面还有一个基于DisCO判别式的理论分析和 **带Cov的"熵-协方差恒等式"**，有点没看懂怎么拆的

### Q1 也就是说QAE只是在GRPO的基础上把mean改成了k百分位数，但是仍然是轨迹级的奖励分配，是一个粗粒度的分配，而不像HEPO或TAPO至少通过token-level的熵来精细地分配梯度
对，QAE 确实只是在 response 级把 mean 换成 quantile,advantage 本身还是整条 response 共享一个标量,没有 token 级精细分配。
- **HEPO/TAPO 的稀疏**:token 级,在一条 response 内部只训部分 token
- **QAE 的稀疏**:response 级,整条 response 要么有信号要么没信号
零 advantage 意味着:这条 response 的 reward 恰好等于 quantile baseline。它既不是"相对好的"也不是"相对差的",组内没有可学的相对信号。
![[Pasted image 20260929150949.png]]

### Q2   QAE只是减掉某个常数，凭啥能让固定比例的advantage变成0呢，万一奖励分布比较不一样的那不就失效了
![[Pasted image 20260929151252.png]]

![[Pasted image 20260929151412.png]]

### Q3 文中说Entropy explosion is disproportionately driven by negative-advantage samples.又说a few high-reward samples can inflate the baseline, turning otherwise competent responses into negative-advantage examples and penalizing useful exploration, which induces entropy collapse.所以negative advantage到底是导致熵坍缩还是爆炸

？？
![[Pasted image 20260929174253.png]]



[35] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen,  Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. Advances in Neural Information Processing Systems, 38:126546–126573, 2026.
似乎有点关系啊，有时间可以看一下