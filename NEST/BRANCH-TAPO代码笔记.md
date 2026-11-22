[GitHub - Labbbbbbbbb/NEST · GitHub](https://github.com/Labbbbbbbbb/NEST)
新增文件：
- 训练入口(/data/home/zdhs0200/NEST/scripts/train_tapo.sh)
- 训练参数(/data/home/zdhs0200/NEST/examples/tapo_trainer/run_alfworld.sh)
- TAPO 核心算法(/data/home/zdhs0200/NEST/tapo/core_tapo.py)
- 算法与运行说明(/data/home/zdhs0200/NEST/tapo/README.md)

TAPO 路径已移除 GiGPO 的观察分组、step 折扣回报和 step 优势叠加，替换为成功/失败优势、固定参考模型概率比值、token 熵筛选及裁剪损失。原 GiGPO 文件保留供旧入口使用。


输出token的熵`entropy` 本身是在 [dp_actor.py (line 395)](/data/home/zdhs0200/NEST/TAPO/verl/workers/actor/dp_actor.py:395) 调用 `_forward_micro_batch()` 时算出来的，然后在（line408）左右
```
pg_loss, tapo_metrics = compute_tapo_loss(
```
的地方将`entropy`传入，通过与阈值比较的`bool`掩膜得到熵掩膜