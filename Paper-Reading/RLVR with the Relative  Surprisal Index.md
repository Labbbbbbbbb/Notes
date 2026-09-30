**"However, existing token-selection criteria typically rely on a single signal, such as entropy, gradient magnitude, KL divergence, or logit-rank shift, leaving open how to jointly characterize predictive uncertainty and the probability of the selected token."**

### Q1 低概率为什么会引入不稳定，虽说低概率的token可能会有很大的概率提升，但ratio超过范围的会被clipped的啊
 clip 解决不了低概率 token 的几个根本问题：
 1.梯度方差爆炸，因为对log求梯度本身分母就会有$\pi_\theta$这个概率值，概率低的token梯度方差会很大，clip不解决这一点
 2.低概率 token 天然容易触发 **off-policy 漂移**: 低概率 token 的**相对变化天然大**——从 0.01 到 0.02 是翻倍，从 0.9 到 0.95 只涨 5.6%。clip 虽然截断了梯度，但**截断可能发生在 ratio 已经很大之后**，参数空间已经发生了显著偏移。
 
有兴趣也可以去看一下那篇⁨Do not let lowprobability tokens over-dominate in rl for llms. In 2nd AI for Math Workshop@ ICML, 2025.