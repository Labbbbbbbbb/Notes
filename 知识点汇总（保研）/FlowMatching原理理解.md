**FlowMatching**相比于**Diffusion**最不同的地方在于它是**ODE**而非**SDE**
$$
dx=v(t)dt
$$
然后对于中间的过程$x_t$比较常见的处理方式是线性插值
$$
xt_​=(1−σ)x_0​+σϵ,ϵ∼N(0,I)
$$
此处$\epsilon$代表$x_1$，是白噪声，遵循diffusion式的从1到0的去噪方向
训练Flow Matching的过程：
```
import torch

# sample
x0 = get_data_sample()          # 真实数据
x1 = torch.randn_like(x0)       # x1 ~ N(0,1) 纯噪声

# 随机采样时间步
t = torch.rand(x0.shape[0])     # t ~ Uniform(0,1)
t_ = t.view(-1, 1, 1, 1)        # reshape 用于广播

# 插值构造 xt
xt = t_ * x1 + (1 - t_) * x0   # 从x0到x1的线性插值

# 目标速度场（ground truth）
vt = x1 - x0                    # dx_t/dt，与t无关

# 模型预测
model_output = model(xt, t)     # 输入带噪图和时间步

# Loss
loss = ((vt - model_output) ** 2).mean()

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()
```




## 统一到SDE形式

有时为了给流模型增加随机性，会将其变为SDE:
原有ODE：
$$
dx=v(t)dt
$$
变成SDE后：
$$
dx_t=f(x_t)dt+g_tdw
$$ 其中$f(x_t)=v_t(x_t)+\frac{g_t^2}{2}\nabla \log p_t(x_t)$ ，是正向过程中对随机量$w$的推散的补偿，以满足边际分布不变。
而已有了正向SDE扩散过程，接下来就要考虑反向去噪过程：

详见Anderson提出的理论：任何前向SDE都有一个等价的反向SDE，两者共享相同的边际分布
若前向过程如下：(0->1,是加噪过程，也即扩散过程)
$$
dx_t=f(x_t)dt+g_tdw
$$
则反向过程为（1->0，去噪过程）
$$
dx_t=[f(x_t) \underbrace{-g^2_t\nabla \log p_t(x_t)}_{\text {score function correction}}]dt+g_td\bar{w}
$$
其中$score=\nabla \log p_t(x_t)$ 其实于“分数”无关，它是对数似然概率的梯度，在随机扰动$g_tdw$ 将分布推散时，score修正会将其往高概率方向上拉，以维持边际分布不变。注意此时$dt<0$

所以在Flow matching中，反向SDE的表达式为：
$$
dx_t=[v_t(x_t) -\frac{g^2_t}{2}\nabla \log p_t(x_t)]dt+g_td\bar{w}
$$
具体倒线性插值形式的Flow matching：(ps.上面的$f$是正向漂移补偿这里$\bar{f}$是反向的补偿)
$$
dx = \underbrace{\left(v - \frac{g^2}{2} \cdot \frac{x_t - (1-\sigma)\hat{x}_0}{\sigma}\right)}_{\text{漂移项}\bar{f}(x,t)}dt + g\,dW \newline =\bar{f}(x,t)dt+g\cdot \sqrt{|dt|}\cdot\epsilon 
$$ 原因： 线性插值路径为 $x_t = (1-\sigma)x_0 + \sigma\epsilon$，此时 score 函数有解析形式：

$$\nabla \log p_t(x_t) = -\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma^2}$$


又： $\mu = x_t + \bar{f}(x_t, t) \cdot dt$ （去噪过程$x_{t-1}$的均值，就是去掉白噪声$w$）

代入漂移项：

$$f(x_t, t) = v - \frac{g^2}{2} \cdot \frac{x_t - (1-\sigma)\hat{x}_0}{\sigma}$$

所以：

$$\mu = x_t + \left(v - \frac{g^2}{2} \cdot \frac{x_t - (1-\sigma)\hat{x}_0}{\sigma}\right)dt$$
分子    $x_t - (1-\sigma)\hat{x}_0 = x_t - \hat{x}_0 + \sigma\hat{x}_0$

又因为线性插值 $x_t = (1-\sigma)\hat{x}_0 + \sigma\epsilon$，所以：

$$\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma} = \epsilon$$

这就是注意里说的 $\frac{x_t-(1-\sigma)\hat{x}_0}{\sigma} \approx \epsilon$，**本质是把 $x_t$ 里的噪声成分提取出来**。

再利用速度场关系 $v = \hat{x}_0 - \epsilon$ 换掉 $\epsilon$，Flow Matching 的速度场定义为：

$$v = \frac{dx_t}{dt} = x_1 - x_0 \approx \hat{x}_0 - \epsilon$$

（这里 $x_0=\hat{x}_0$ 是干净图，$x_1=\epsilon$ 是噪声）

所以：

$$\epsilon = \hat{x}_0 - v$$

代入第二步：

$$\frac{x_t-(1-\sigma)\hat{x}_0}{\sigma} = \epsilon = \hat{x}_0 - v$$

然后把 $\hat{x}_0$ 用 $x_t$ 和 $v$ 表示出来：把 $\epsilon = \hat{x}_0 - v$ 代回 $x_t$：

$$x_t = (1-\sigma)\hat{x}_0 + \sigma(\hat{x}_0 - v) = \hat{x}_0 - \sigma v$$

所以：
$$\hat{x}_0 = x_t + \sigma v$$
z再把$x_0$代回 $\mu$ 展开
$$\mu = x_t + \left(v - \frac{g^2}{2}(\hat{x}_0 - v)\right)dt$$
$$= x_t + \left(v - \frac{g^2}{2}(x_t + \sigma v - v)\right)dt$$
$$= x_t + \left(v - \frac{g^2}{2} x_t - \frac{g^2}{2}(\sigma-1)v\right)dt$$

整理 $x_t$ 和 $v$ 的系数：

$$\mu = x_t\underbrace{\left(1 + \frac{g^2}{2\sigma} \cdot (-\sigma) \cdot \frac{\sigma}{\sigma}\right)}_{\approx 1+\frac{g^2}{2\sigma}dt} \cdots$$

最终收敛到：

$$\boxed{\mu = x_t\left(1 + \frac{g^2}{2\sigma}dt\right) + v\left(1 + \frac{g^2(1-\sigma)}{2\sigma}\right)dt}$$

于是$x_{t-1} = \mu +g\cdot \sqrt{-\Delta t} \cdot w$ 

### 推导 $\log p(x_{t-1} | x_t)$

由 SDE 的 Euler-Maruyama 离散化，$x_{t-1}$ 给定 $x_t$ 时服从高斯分布：

$$x_{t-1} = \underbrace{\mu}_{\text{均值}} + g\sqrt{-dt} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

所以：

$$p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu, \underbrace{g^2(-dt)}_{\sigma_{noise}^2} \cdot I)$$
#### 代入高斯对数概率密度

$$\log p(x_{t-1}|x_t) = -\frac{|x_{t-1} - \mu|^2}{2\sigma_{noise}^2} - \log \sigma_{noise} - \frac{1}{2}\log(2\pi)$$

其中 $\sigma_{noise} = g\sqrt{-dt}$

#### 对应代码

```python
log_prob = (
    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
    - torch.log(std_dev_t * torch.sqrt(-1 * dt))          # -log(σ_noise)
    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))) # -log(sqrt(2π))
)
```

逐项对应：

|代码项|公式项|
|---|---|
|`-(prev_sample - mean)^2 / (2 * σ_noise^2)`|$-\frac{\|x_{t-1}-\mu\|^2}{2\sigma_{noise}^2}$|
|`-log(std_dev_t * sqrt(-dt))`|$-\log \sigma_{noise}$|
|`-log(sqrt(2π))`|$-\frac{1}{2}\log(2\pi)$|
完整公式：
$$\boxed{\log p(x_{t-1}|x_t) = -\frac{|x_{t-1}-\mu|^2}{2g^2(-dt)} - \log(g\sqrt{-dt}) - \frac{1}{2}\log(2\pi)}$$