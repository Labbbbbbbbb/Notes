主要探讨了CoT中的熵在不同token中的分布模式，探究了RLVR中的动力学规律
### 训练过程中 token 熵应该怎么变?

论文 Section 4 的两个核心发现:

> 1. **RLVR preserves the existing entropy patterns of the base models** — 哪些位置是高熵、哪些是低熵,这个"排名"基本不变
> 2. **RLVR predominantly alters the entropy of high-entropy tokens, whereas the entropy of low-entropy tokens varies only within a small range** — 低熵 token 的熵几乎不动,只有高熵 token 的熵在变
> 3. Tokens with higher initial entropy tend to experience greater entropy increases after RLVR，最高熵的那部分的熵反而增加

但是高熵的token的熵随着训练进行应该是缓慢地下降，并且最后相对于低熵token也是维持在较高的水平的，**熵的"排名模式"没变**——这个位置仍然是个三岔路口(高熵 token 位置不会变成低熵)。（uring RLVR training, the reasoning model largely preserves the base model’s entropy patterns, showing only gradual and minor changes.）但**熵的"分布形状"在变**:概率质量在向好的分支集中,但保持足够的平坦度来保留探索能力。
熵是用来决定要不要给梯度的门控或者乘在advantage上用来调节梯度大小的因子，而优化本身并非优化熵的值，并不是要把所有高熵都降至确定，优化的目标是调整分布中top-k候选的概率比

之所以说选取20%subset的高熵token是“facilitate exploration”的，是因为全token梯度的隐藏危害是低熵 token 在压塌整体熵，低熵token本身分布比较尖，倾向于分布得更“尖”，从而抑制探索，而高熵token分布比较平，梯度调制会让它改变top-k的分布而不是倾向于确定，不会让熵快速坍缩。


优化的策略梯度公式：
![[Pasted image 20260929021544.png]]
式子中dataset D中的(q,a)对是离线的数据，代表问题和标准答案，但是$0^i$是由policy rollout出来的，梯度作用在这些过程上，是一个online的更新过程。其中优势A的计算如下
![[Pasted image 20260929021910.png]]
is_equivalent是语义上的判分器，这也是RLVR中V的lai

### Q1
RLVR中的token永远都是自己采的，并不存在一个离线的token库，那么衡量一个token的熵的时候肯定是用当下的policy、基于当时的输入的前序token才能得到生成该token的熵，这在策略梯度的时候怎么能随时衡量哪个token的熵排在前80%？又怎么能直接给出高熵token的词云图（在没说输入的prompt和前序token是啥的情况下）？
```
阶段 A:Rollout(冻结策略快照 θ_old)
  for 每个 prompt × G 条 response:
      for t = 1 ... T:
          p_t = Softmax(forward(θ_old; o_<t))
          o_t ~ p_t
          H_t = -Σ p_t log p_t        # 顺手缓存
  # ↓ 此时所有位置的熵都已到手,排序才发生
  threshold = 0.672                    # 见下文
  m_t = 1[H_t > threshold]            # 每个位置独立判断,无需全局实时排序

阶段 B:Update(梯度更新,可能多个 micro-step)
  L = Σ_t m_t · PG(ρ_t, Â_t) / Σ_t m_t
```
**阈值根本不是每批动态算的——论文用的是一次性估计的全局绝对阈值**

> hthreshold​=0.672 ... estimated by calculating the **80th percentile among the sampled 106 tokens**

即论文先在 106 个采样 token 上做一次分位数估计,得到一个**固定绝对阈值 0.672**,整个训练期间每个位置只做一次独立的大小比较。连"批量排序"都不需要——这依赖的正是"熵模式跨 prompt、跨训练阶段稳定"这一发现,所以全局阈值合法。

(其他实现里也有 per-batch 池化分位数或 per-sequence 内取 top 20% 的变体,论文选择了最简单的全局阈值。)

严格地说,**不存在"一个 token 的熵"这种东西,只存在"某个 context 下某个位置的分布的熵"**。词云的构造方式:

1. 在**整个基准集**(六个 benchmark 的全部 prompt,每条 16 个 response)上 rollout,得到上百万个带 Ht​ 的位置
2. 取出所有满足 Ht​>hthreshold​ 的位置
3. 把这些位置上**实际采样到的 token id 解码成字符串**,做频次统计
4. 词云里字号 ∝ 该字符串出现在高熵位置的频次(更严谨的版本会除以它的全局频次,得到 lift/富集倍数)

所以词云回答的问题不是"'Wait' 这个词熵高",而是:

> **"在跨大量 prompt、大量前缀的统计意义上,哪些 token 字符串倾向于在高熵位置被采样到?"**

prompt 和前缀并没有缺席——它们是整个评测语料库,只是聚合后从图上消失了。