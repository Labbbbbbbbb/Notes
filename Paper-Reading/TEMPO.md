# QA
1.文中提到：Recent “key-token ideas” (Wang et al., 2025) move toward finer signals, but they do not use the implicit prefix tree that multiple responses to the same prompt already define.
也就是Beyond the 80/20 rule，所以这篇文章是TEMPO的无前缀树版本吗？
# 整理

TEMPO的优势计算公式是
$$
\begin{equation}
\hat{A}_{i,t} = \frac{1}{\mathrm{std}(r)} \left[ \underbrace{r_i - \mathrm{mean}(r)}_{\text{MC signal}} + \underbrace{V(s_{t+1}) - V(s_t)}_{\text{TD error}} \right]
\end{equation}
$$
而不是像我之前以为的直接用树的叶子节点获得的价值来当梯度，一个是因为Advantage的估计方式就是$Q-V=r+\gamma*V_{t+1}-V_t$，在$\gamma=1$的情况下就是$r+V_{t+1}-V_t$，符合优势的计算逻辑
二来加法是为了避免两个正交维度（？）的优势相乘导致正负号的问题
三来这里的$V(s_0)$ 其实就是$mean(r)$，和$r_i-mean(r)$在量纲上是完全相同的，因此先加法再一起除std(r)，而使用TD形式可以避免处在好分支上的普通token也同样获得高优势。
![[Pasted image 20260921174158.png]]