[Stepwise Intrinsic Rewards for Reasoning in Large Language Models](https://arxiv.org/pdf/2602.01034)
![[Pasted image 20261021122615.png]]


![[Pasted image 20261021122630.png]]

>[!NOTE] Idea
>显然在ALFWorld并没有一个标准答案来指示每一个step的标准答案是什么。但是可以考虑一下是不是可以用PDDL等等之类的规则，只为最终成功的步骤来进行token级别的归因（此时成功的步骤本身就是标注，可以计算每个token出来之后计算得到这个答案的概率），可以为失败的组新建一个分组，比如查证得是PDDL哪个环节出问题的回答分为一个组去查看哪个token导致了这个错误

