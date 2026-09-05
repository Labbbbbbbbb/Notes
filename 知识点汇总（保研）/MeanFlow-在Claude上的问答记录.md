https://claude.ai/share/2f348716-6891-4b4e-b927-1ba0d8de0fd9


# MeanFlow 论文阅读笔记

> 整理自今日阅读讨论，涵盖 Consistency Models、Flow Matching、Diffusion ODE、EDM、MeanFlow 等核心概念。

---

## 1. Consistency Models 是什么

![Consistency Models 原文段落](https://claude.ai/chat/1775552037315_image.png)

### 背景

Flow Matching 和 Diffusion Models 在生成时都需要**迭代采样（iterative sampling）**，即从噪声出发经过几十乃至上百步去噪才能生成图像，速度较慢。Consistency Models 的目标是：

> **用极少的步骤（理想情况下一步）完成生成。**

### 核心机制：一致性约束（Consistency Constraint）

在同一条生成路径（trajectory）上采样的不同时间点，网络应该给出**相同的输出**（都映射到同一个干净数据点）。这样无论从路径上哪个噪声级别出发，模型都能一步直接预测最终结果。

### 存在的问题

|问题|说明|
|---|---|
|约束施加在网络行为上|是作为网络的"行为性质"强制加入的，缺乏理论推导|
|底层真实向量场未知|应指导学习的 ground-truth field 性质不清楚|
|训练不稳定|缺乏理论支撑，容易不稳定|
|需要离散化课程|必须精心设计 discretization curriculum，逐步收窄时间域|

---

## 2. 最优解与具体网络无关

![最优解独立于网络](https://claude.ai/chat/1775552779121_image.png)

### Consistency Models 的问题

一致性约束是**强加在网络行为上的**，没有一个独立于网络存在的"真实目标"。网络的目标依赖于网络自己的输出，形成自举（bootstrapping）循环，导致训练不稳定。

### 本文的做法

训练网络去拟合 **average velocity field（平均速度场）**，这个场是**客观存在的**，由数据分布本身决定，与网络结构无关。

|场景|类比|
|---|---|
|Consistency Models|标准答案由学生自己的答卷动态生成|
|本文方法|标准答案客观存在，任何学生都在逼近同一个答案|

> **结论**：损失函数的优化景观是稳定的，梯度方向始终指向固定目标，不存在"目标随训练漂移"的问题。

---

## 3. Diffusion Model 的 ODE 版本

![Diffusion ODE 原文](https://claude.ai/chat/1775553523688_image.png)

### 原始 Diffusion：SDE 视角

标准扩散模型的加噪/去噪过程是一个**随机微分方程（SDE）**：

$$dx = f(x,t)dt + g(t)d\mathbf{w}$$

其中 $d\mathbf{w}$ 是随机噪声项，每次采样路径都不一样。

### Probability Flow ODE

Song Yang 等人证明：对于任意扩散过程（SDE），都存在一个对应的**确定性 ODE**，使得两者在每个时刻的**边缘分布完全相同**：

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$

### SDE vs ODE 对比

|特性|SDE（原始）|Probability Flow ODE|
|---|---|---|
|随机性|每次路径不同|确定性轨迹|
|边缘分布|$p_t(x)$|完全相同的 $p_t(x)$|
|采样速度|较慢|可用高阶ODE solver加速|
|可逆性|❌|✅ 严格可逆|
|似然计算|困难|✅ 可精确计算|

### 意义

- **加速采样**：可使用 DDIM、DPM-Solver 等高阶 ODE 求解器大幅减少步数
- **精确编码**：可逆性使图像编辑成为可能
- **与 Flow Matching 的联系**：Flow Matching 本质上是在 ODE 框架下直接建模速度场

---

## 4. Few-step 方法：蒸馏与 Consistency

![Few-step 方法](https://claude.ai/chat/1775553729678_image.png)

### 将多步扩散模型蒸馏为少步模型

用已训练好的 **Teacher 模型**监督训练更快的 **Student 模型**：

```
Teacher: 需要1000步的扩散模型
              ↓  蒸馏
Student: 只需4步/1步的模型
```

以 Progressive Distillation 为例：Teacher 走 2 步，Student 学会用 1 步走到同样位置，每轮步数减半。

### Score Distillation（SDS）

来自 DreamFusion，目标不同——用预训练扩散模型中蕴含的 **score（梯度信息）** 来指导另一个参数的优化：

$$\nabla_\theta \mathcal{L} = \mathbb{E}\left[\omega(t)\left(\hat{\epsilon}_\phi(x_t, t) - \epsilon\right)\frac{\partial x}{\partial \theta}\right]$$

**应用场景**：文生3D、图像编辑、风格迁移等。

### Distillation-based Methods 总览

|方法类型|核心思路|目标|
|---|---|---|
|Progressive Distillation|步数逐步减半蒸馏|加速采样|
|Consistency Distillation|从Teacher轨迹学习一致性|单步生成|
|Score Distillation (SDS)|用score梯度优化外部参数|3D/编辑生成|
|Distribution Matching Distillation|匹配生成分布而非轨迹|加速+质量|

### Rectified Flow 与蒸馏的关系

Rectified Flow **本身不是蒸馏方法**，是独立的生成模型训练框架，直接回归直线路径的速度：

$$\mathcal{L} = \mathbb{E}\left[| v_\theta(x_t, t) - (x_1 - x_0) |^2\right]$$

**Reflow** 操作形式上像蒸馏（用前一轮模型生成配对数据再训练），但目的是让轨迹越来越直。

---

## 5. Conditional Flow vs Marginal Flow

![Flow 类型对比图](https://claude.ai/chat/1775564131761_image.png)

### 左图：Conditional Flow（条件流）

同一个中间点 $z_t$ 可能来自不同的 $(x, \epsilon)$ 配对，导致**同一输入对应多个速度方向**（一对多映射），网络无法直接学习。

### 右图：Marginal Flow（边际流）

对所有经过 $z_t$ 的路径做期望：

$$v(z_t, t) = \mathbb{E}[v_t \mid z_t]$$

每个点只有**唯一确定的速度向量**，网络可以学习。

### 关键对比

||Conditional Flow|Marginal Flow|
|---|---|---|
|速度定义|单条路径上的瞬时速度|所有路径的期望速度|
|同一 $z_t$ 的速度|多个（一对多）|唯一（一对一）|
|能否直接作为训练目标|❌|✅|
|路径形状|各条路径是直线|平均后的场是弯曲的|

---

## 6. MeanFlow 训练目标

![MeanFlow 训练公式](https://claude.ai/chat/1775635876763_image.png)

### 损失函数

$$\mathcal{L}(\theta) = \mathbb{E}|u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}})|_2^2 \tag{9}$$

$$u_{\text{tgt}} = v(z_t, t) - (t-r)\left(v(z_t,t)\partial_z u_\theta + \partial_t u_\theta\right) \tag{10}$$

### 从 Marginal Velocity 到 Conditional Velocity

将 $v(z_t, t)$（边际速度，不可直接计算）替换为 $v_t$（条件速度，可直接计算）：

$$u_{\text{tgt}} = v_t - (t-r)\left(v_t \partial_z u_\theta + \partial_t u_\theta\right) \tag{11}$$

**替换依据**：$v(z_t,t) = \mathbb{E}[v_t \mid z_t]$，用条件速度的无偏估计替换边际速度，期望损失不变。类似随机梯度下降用单样本梯度估计全局梯度。

||$v(z_t, t)$（Eq.10）|$v_t$（Eq.11）|
|---|---|---|
|含义|边际速度（期望）|条件速度（单样本）|
|是否可直接计算|❌ 需要对所有路径积分|✅ 由 $(x,\epsilon)$ 直接给出|
|关系|$v(z_t,t) = \mathbb{E}[v_t\|z_t]$|是前者的无偏估计|

---

## 7. JVP 计算与 Stop-Gradient

![JVP 计算](https://claude.ai/chat/1775661858206_image.png)

### Algorithm 1: MeanFlow 训练流程

```python
# fn(z, r, t): function to predict u
# x: training batch

t, r = sample_t_r()
e = randn_like(x)

z = (1 - t) * x + t * e
v = e - x

u, dudt = jvp(fn, (z, r, t), (v, 0, 1))

u_tgt = v - (t - r) * dudt
error = u - stopgrad(u_tgt)

loss = metric(error)
```

### 为什么 JVP 只需一次 backward pass？

**关键：stopgrad 切断了二阶梯度。**

正常情况下若 $\frac{d}{dt}u$ 出现在 loss 里且需要对 $\theta$ 求梯度，需要**二阶导数（double backpropagation）**，非常昂贵。

但 `dudt` 被放入 `stopgrad(u_tgt)` 中，对 $\theta$ 求梯度时被当作常数：

```
① jvp 前向计算 dudt（一次前向+切线传播）
② 计算 loss = metric(u - stopgrad(u_tgt))
③ 对 loss 关于 θ 做普通 backward（一次反向）
```

### JVP 各方向的维度

```python
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
#                              ↑  ↑  ↑
#                           d维  1维 1维  ← 切线向量的维度
```

- $\partial_z u$ 的切线 $v_t \in \mathbb{R}^d$：因为 $z$ 是图像空间，维度为 $d$（$32\times32\times4$）
- $\partial_r u$、$\partial_t u$ 的切线为标量：因为 $r, t$ 是时间变量

---

## 8. CFG 的边际等价性

![CFG 等式](https://claude.ai/chat/1775710748715_image.png)

### 公式

$$v^{\text{cfg}}(z_t, t) \triangleq \mathbb{E}_\mathbf{c}[v^{\text{cfg}}(z_t, t \mid \mathbf{c})] = \omega, \mathbb{E}_\mathbf{c}[v(z_t, t \mid \mathbf{c})] + (1-\omega), v(z_t, t) = v(z_t, t)$$

### 含义

把 CFG 速度场对所有条件 $\mathbf{c}$ 取期望后，结果退化回**无条件的边际速度场**，与 guidance 强度 $\omega$ 无关。

**意义**：CFG 只是在条件层面重新分配概率流方向，不改变整体边际统计特性。这为 MeanFlow 自然纳入 CFG 提供了理论依据。

---

## 9. EDM Preconditioning

![EDM Preconditioning](https://claude.ai/chat/1775711041449_image.png)

### EDM 是什么

**EDM = Elucidating the Design Space of Diffusion-Based Generative Models**（Karras et al., NeurIPS 2022）

核心贡献：把各种扩散模型统一到同一框架下，对每个设计选择做消融实验，找出最优组合。

### Preconditioning 的必要性

不同噪声级别 $\sigma$ 下，输入输出的数值范围差异极大：

```
σ 很小（接近干净图）：输入幅度 ~ σ_data ≈ 0.5
σ 很大（几乎全是噪声）：输入幅度 ~ σ >> 1
```

Preconditioning 通过三个系数解决这个问题：

$$D_\theta(x, \sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot F_\theta(c_{\text{in}}(\sigma) \cdot x, \sigma)$$

|系数|公式|作用|
|---|---|---|
|$c_{\text{in}}$|$\frac{1}{\sqrt{\sigma_{\text{data}}^2 + \sigma^2}}$|归一化输入方差为1|
|$c_{\text{out}}$|$\frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$|控制网络输出幅度|
|$c_{\text{skip}}$|$\frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}$|跳跃连接：σ小时偷懒，σ大时努力|

---

## 10. MeanFlow 网络实现：各变量含义

### 坐标变换（Flow → EDM）

$$\sigma = \frac{t}{1-t}, \qquad x = \frac{z_t}{1-t}$$

验证：$\frac{z_t}{1-t} = \frac{(1-t)x_0 + t\epsilon}{1-t} = x_0 + \sigma\epsilon$ ✅

### 完整前向流程

$$z_t, t \xrightarrow{\text{坐标变换}} \sigma, x \xrightarrow{c_{\text{in}}} \text{UNet} \xrightarrow{F_x} \xrightarrow{c_{\text{skip}}, c_{\text{out}}} D_x \xrightarrow{\text{转回Flow}} u$$

|变量|含义|
|---|---|
|$\sigma$|EDM 噪声强度，$= t/(1-t)$|
|$x$|EDM 坐标下的带噪图像，$= z_t/(1-t)$|
|$c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}$|归一化缩放系数|
|$F_x$|UNet 原始输出（中间量）|
|$D_x = c_{\text{skip}} x + c_{\text{out}} F_x$|对干净图像 $x_0$ 的预测|
|$u = (z_t - D_x)/t$|平均速度场（最终输出）|

### 最后一步的直觉

$$u = \frac{z_t - D_x}{t} = \frac{\text{当前位置} - \text{预测终点}}{\text{剩余时间}}$$

> "我现在在 $z_t$，预计终点是 $\hat{x}_0$，平均速度是多少"

---

## 11. EDM2 UNet：原始 vs MeanFlow 版本

### 两个 forward 的对比

**唯一区别：时间嵌入**

```python
# 原始 EDM2：单时间嵌入
emb = self.emb_noise(self.emb_fourier(noise_labels))

# MeanFlow 版本：双时间嵌入，等权混合
emb = mp_sum(
    self.emb_noise_t(self.emb_fourier(sigma_t)),
    self.emb_noise_r(self.emb_fourier(sigma_r)),
    t=0.5
)
```

### 为什么需要两个时间变量？

MeanFlow 的网络预测 $u_\theta(z_t, r, t)$，需要同时知道：

- $t$（我现在在哪，当前噪声级别）
- $r$（我要去哪，目标噪声级别，通常 $r=0$）

### 为什么可以复用 Encoder/Decoder 权重？

|部分|能否复用|原因|
|---|---|---|
|Encoder/Decoder 主体|✅ 可以|特征提取能力与预测目标无关|
|时间嵌入层|❌ 需要重新学|从单时间变成双时间，结构改变|
|输出层|⚠️ 可以微调|输出量纲相同但目标分布略有差异|

两个任务（预测瞬时速度 vs 平均速度）的**特征层面高度相似**：都需要理解图像语义结构、感知噪声级别、提取多尺度特征。且：

$$u(z_t, r, t) \xrightarrow{r \to t} v(z_t, t)$$

平均速度是瞬时速度的积分平均，两者高度相关，底层特征通用。

---

## 总结：MeanFlow 的核心创新

```
标准 Flow Matching        →    训练瞬时速度场 v(z_t, t)
                                需要多步 ODE 求解器采样

MeanFlow                  →    训练平均速度场 u(z_t, r, t)
                                单步直接到达目标：z_r = z_t - (t-r) · u
```

**三个关键设计：**

1. **MeanFlow Identity**：推导出客观存在的 ground-truth 目标场，训练稳定
2. **JVP + stopgrad**：高效计算 $\frac{d}{dt}u$，避免二阶梯度，仅增加 <20% 训练开销
3. **复用 EDM2 结构**：仅修改时间嵌入，其余权重可复用/微调




## Representation Collaspe
---

Meanflow本质上还是一个ODE，它是否存在在模型训练过程中随机性不足的问题？是否更容易导致表征坍缩？

## 什么是表征坍缩（Representation Collapse）

在生成模型语境下，表征坍缩指：

> 模型的输出**丧失多样性**，倾向于映射到少数几个"安全"的输出，忽略输入的细微差异。

---

### MeanFlow 有哪些机制容易诱发坍缩

**① 平均操作本身的平滑效应**

MeanFlow 的核心是对速度场做时间平均：

$$u(z_t, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau), d\tau$$

**平均天然是一个平滑算子**，它会：

- 抹去轨迹上的局部波动
- 让不同起点的轨迹在平均意义下趋于相似
- 模型有动机学到"中庸"的表征

这与 VAE 中的**posterior collapse**机制类似——模型发现输出均值比输出细节代价更低。

---

**② ODE 确定性 + 回归目标的联合压力**

MeanFlow 用回归损失训练：

$$\mathcal{L} = | f_\theta(z_t, t) - \text{target} |^2$$

MSE 损失的最优解是**条件期望**，即：

$$f^*(z_t) = \mathbb{E}[\text{target} \mid z_t]$$

当多个不同的真实输出对应同一个 $z_t$ 时，模型会学到它们的**均值**，而不是任何一个具体的输出——这正是坍缩的典型形式。

---

**③ 自举（Bootstrap）目标的坍缩风险**

前面提到 MF 的 target 依赖网络自身，这类自举机制在对比学习中是著名的坍缩诱因：

```
网络输出 → 生成 target → 网络拟合 target
    ↑_________________________________|
```

如果没有足够的约束（如对比损失、stop-gradient、EMA），网络可能找到**平凡解**：把所有输入都映射到同一个点，损失照样为零。

---

### 与其他方法对比

|方法|坍缩风险|原因|
|---|---|---|
|**扩散模型（SDE）**|较低|每步随机噪声防止轨迹退化|
|**Consistency Models**|中等|相邻步一致性约束有一定保护|
|**MeanFlow**|相对较高|平均操作 + 回归目标 + ODE确定性三重叠加|
|**VAE**|中等|KL项显式防止posterior collapse|

---

### 但也有抑制坍缩的因素

当然 MeanFlow 并非没有保护机制：

- **输入噪声的多样性**：不同的 $z_t$ 仍然提供了区分信号
- **时间步条件化**：$(z_t, r, t)$ 的组合空间很大，难以完全坍缩
- **Flow Matching 预训练目标的约束**：教师模型提供的 fixed target 在一定程度上锚定了输出空间

---

### 一句话总结

MeanFlow 由于**平均操作的平滑效应、MSE回归目标趋向条件均值、以及ODE确定性**三者叠加，理论上比 SDE 类方法更容易出现表征坍缩的倾向；但实践中能否真正坍缩，还取决于训练细节、数据多样性以及是否有足够强的锚定信号。
