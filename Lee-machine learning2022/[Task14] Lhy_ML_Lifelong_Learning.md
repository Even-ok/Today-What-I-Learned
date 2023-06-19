## [Task14] Lhy_ML_Lifelong_Learning

> 我特别喜欢这个词，叫“终身学习”，我希望我也可以做到这样的一个境界！



**Q1: 终身学习的大致流程？**

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619100910188.png" alt="image-20230619100910188" style="zoom:67%;" />

先从旧任务的标记数据中学习，然后使用线上的数据资料更新模型参数。



**Q2: 终身学习会遇到的问题——different domain, catastrophic forgetting, order(顺序)**

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619101235339.png" alt="image-20230619101235339" style="zoom:67%;" />

比如说这个，学了任务2之后，机器可能会遗忘了对任务1所学到的知识。



**Q3: 多任务学习（multi-task learning）和终身学习（lifelong）的区别？**

multi-task learning会把以前学到的东西，在新的阶段还拿出来，一块和新数据训练（相当于机器重新复习了），所以multi-task learning通常是LLL的upper bound。

如果全部倒在一起训练，这样十分**难以存储**，**训练时间长**。



**Q4: 终身学习（持续学习）的目的：**

不同任务间的资料互通有无（一个模型可以解决大多数的任务）

解决存储和训练时间的问题



**Q5: transfer learning 和 lifelong learning的区别**

- transfer注重于第一个任务学到的东西，对第二个任务有没有什么大的帮助（关注点在**第二个任务**）
- lifelong learning关注点在，学习完第二个任务之后再**回头看**第一个任务，还能不能有很好的解决效果？



**Q6: 对LLL的评估方式**

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619102549153.png" alt="image-20230619102549153" style="zoom:50%;" />



**Q7: 当前终身学习的3种解决方案**

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619102728775.png" alt="image-20230619102728775" style="zoom:50%;" />



**Q8: 灾难性遗忘的原因**

不同的任务有不同的error surface

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619102807731.png" alt="image-20230619102807731" style="zoom:50%;" />



**Q9: regularization-based主要是在损失函数上怎么改进的？**

就是加了一个惩罚项，然后用$b_i$来衡量每一个参数的重要性，希望后面学的参数和前面学的参数尽量地接近（在某些参数上）   常见的是**人去设定**的$b_i$

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619103002228.png" alt="image-20230619103002228" style="zoom:50%;" />



**Q10: $b_i$设置的两种极端会带来什么问题？**

- 如果希望各个任务之间互不影响，公式里面的$b_i$设置为0，那个就会有catastrophic forgetting的问题
- 如果$b_i$设置的很大，希望尽可能和之前学习的参数保持一样，就会有Intransigence的问题（对新的任务学不好）
- example:

<img src="C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230619103044896.png" alt="image-20230619103044896" style="zoom:50%;" />



**Q11: 方法介绍（from homework）**

- **EWC**

  > Our approach remembers old tasks by selectively slowing down learning on the weights important for those tasks. 
  >
  > The mammalian brain may avoid catastrophic forgetting by protecting previously-acquired knowledge in neocortical circuits
  >
  > Slows down learning on certain weights based on how important they are to previously seen tasks
  >
  > and can therefore be imagined as a spring anchoring the parameters to the previous solution, hence the name **elastic**
  >
  > In the EWC algorithm, the definition of the loss function is shown below:
  >
  >  $$\mathcal{L}*_B = \mathcal{L}(\theta) + \sum_*{i} \frac{\lambda}{2} F*_i (\theta_*{i} - \theta_{A,i}^{*})^2  $$
  >
  > The definition of $F$ is shown below.
  >
  > $$ F = [ \nabla \log(p(y_n | x*_n, \theta_*{A}^{*})) \nabla \log(p(y_n | x*_n, \theta_*{A}^{*}))^T ] $$
  >
  > We only take the diagonal value of the matrix to approximate each parameter's $F_i$.

其实也就是用可导参数的一阶导数的平方，再做均值化，作为那个precision matrix中对应于这个可到参数的guard! 

这个EWC我不太会计算，但是我找到了一个学姐的博客！ 写的超级好！  要多多向学姐学习，搞好数学！

[Continual Learning 笔记: EWC / Online EWC / IS / MAS - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/205073566)

终于知道为什么EWC只取参数一阶导数的平方了，是因为泰勒展开后，因为有最大值，一阶导为0，那么考虑二阶导。二阶导需要减少计算开销，所以转成了Fisher信息矩阵，可以通过一阶导数的平方计算出来！

> 我认为的另一个理解方式是，Fisher 信息矩阵也反映了我们对参数估计的不确定度。二阶导越大，说明我们对该参数的估计越确定，同时 Fisher 信息也越大，惩罚项就越大。于是越确定的参数在后面的任务里更新幅度就越小。



- **MAS**





------

### Some thought

当看到自己的学长学姐都这么优秀的时候，我就会感觉目前自己的知识、经历、眼界是远远不足的。有的时候我可能会为一些小的荣誉沾沾自喜，但是在见识到优秀的前辈们如此脚踏实地，有着如此丰富的学识与经历却并未过度强调或过度追求那些所谓的奖项和荣誉时，我就会觉得受之有愧。

Anyway，目前我的目标不仅在补好基础知识，还要大力减肥！！！:facepunch:

:link:[LLL colab](https://colab.research.google.com/drive/1QzyvUSwa_8d93jJTONX4I6Pqn9E2ARYs#scrollTo=7OTZLwxrWFbL)