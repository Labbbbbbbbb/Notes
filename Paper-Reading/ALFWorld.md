[GitHub - alfworld/alfworld: ALFWorld: Aligning Text and Embodied Environments for Interactive Learning · GitHub](https://github.com/alfworld/alfworld),ICLR2021

### ALFWorld框架
ALFWorld的主要贡献是贡献了一个Text和仿真物理世界对齐的benchmark，并且在这个benchmark中 用DAgger训练了一个agent(Butler)。
这样的benchmark使得机器人不再只能用任务轨迹学习（尽管它也提供了轨迹训练数据），更可以通过on-policy的rollout，使用DAgger增强数据或者直接使用RL。

>[!NOTE] ALFWorld框架
>ALFWorld从ALFRED已有的任务定义和轨迹标注中抽取符号状态，再用PDDL作为中间表示，最后分别生成TextWorld和THOR两个环境。

```
             PDDL规则
                 |
                 |
       ----------------------
       |                    |
       ↓                    ↓

 TextWorld              AI2-THOR 【提供视觉输入，为高层命令提供底层执行】

符号状态               视觉物理状态

```

给一张GPT巨献
![[Pasted image 20261015113312.png]]

TextWorld输出流程：
![[Pasted image 20261015111817.png|598]]

![[Pasted image 20261015111920.png|547]]

### 环境搭建小提示（提示我自己）：
主机上用的是源码下载模式，没有用教程中的虚拟py环境，直接用conda
```
conda activate alfworld
```
### 超低自由度
尽管ALFWorld中训练的agent是支持按token序列生成，但是ALFWorld只接受非常固定的动作集的输入，textworld提供统一的`infos["admissible_commands"]`。这也是因为整个ALFWorld都是基于规则规划的，不存在真正的语义理解，无法接受任意自然语言的输入。
之前已经遇到过这样的情况：
```
> take creditcard 3
Nothing happens.

> put creditcard 3 in/on armchair 1
Nothing happens.
```
ALFWorld 的文本环境经常用统一的 `Nothing happens.` 表示动作没有成功执行




### THOR运行示例
按照官方教程，运行
```
$ alfworld-play-thor $ALFWORLD_DATA/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --controller oracle_astar --debug
```
![[Pasted image 20261015104623.png]]
带着AI2Thor一起运行后如右图小框所示，随着终端命令的输入这个场景会同时一起变换

![[Pasted image 20261015105510.png]]

```
cmd = prompt('> ', ...) #输入文本
agent.step(cmd)  #执行，环境变化
```
但输入的是高层语义动作，底层导航会由 Oracle A* Controller 自动完成

让我感到疑惑的是如果cmd能直接给到thor然后让底层规划器执行、完成环境变化，那么于此同时还有一个文本环境的变化，在这个过程里似乎没有像构建的时候那样先用一个统一的pddl来构建然后分别渲染，而是文本自己变化，thor自己变化，拷打GPT结果如下，似乎只是靠构建时两者本身就是通过pddl对齐的，所以在这个变换的时候也是可以自己保持一致的，毕竟都是基于规则的变化，写死的话应该不会飘。
![[Pasted image 20261015112021.png|505]]
>TextWorld 和 AI2-THOR 所共享的物体类别、任务目标和高层动作语义，在 Benchmark 构建阶段已经由开发者建立了对应关系。
>ALFWorld 是通过从 ALFRED 物理任务中提取初始符号状态，再配合统一的高层动作语义及人为实现的 THOR 控制器，建立 TextWorld 与具身环境的对应关系。TextWorld 由 PDDL 引擎更新，THOR 由物理引擎更新；两者追求语义一致，但没有严格的实时状态同步保证。

### BUTLER训练流程
![[Pasted image 20261015110639.png]]

**Brain 输出文本动作后，Body 负责物理执行**
Brain 仍然输出：
```
take potato 1 from countertop 2
```
但 AI2-THOR 并不能直接执行这个字符串，因此，Body 需要将它转换为具体操作，例如：
1. 确定 `countertop 2` 对应的仿真环境物体。
2. 通过导航模块移动到合适的交互位置。
3. 定位 `potato 1` 对应的物体。
4. 调用 AI2-THOR 的拾取操作。
5. 读取环境执行结果。
原论文使用 A* 导航等现成模块完成这部分工作。

![[Pasted image 20261015111337.png|525]]
### DAgger
![[Pasted image 20261015105205.png|398]]
在 ALFWorld 中，DAgger 的专家可以是基于环境规则和规划机制构建的 handcoded expert。例如：
```
infos["extra.expert_plan"]
```
可以为当前状态提供专家规划动作。
