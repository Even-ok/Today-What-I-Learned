## 失踪人口回归之continual/lifelong learning presentation draft



> *前言：这段时间失踪是因为看了10+的traditional/pre-trained系列的连续学习/终身学习方面的文章，虽然是摆烂着看的，但也还是咬着牙看完了。基于下周要做的一个组会PPT，在这里先做一下文字版的解读。虽然我还是没有读懂一个baseline的代码，以及把毕设的promote工作拖延了很久。。。 搞完这个PPT咱们就去研究毕设的incremental工作~*



### Definition

持续/终身学习（下面简称持续学习）就是让模型从连续信息流中学习的一种方法，就是训练数据不会一次性提供，而是会一批一批按照顺序提供，让模型渐进式地学习且适应这些数据。

在持续学习中需要注意两方面的问题：

- 把之前学到的知识拓展到新任务/新数据上，也就是**对新知识学的好 **  --> 可塑性

- 保持对于旧任务的学习能力，也就是**不忘记旧知识**    ---> 稳定性

  

**continual learning vs. incremental learning vs. lifelong learning**

三者如果不作严格区分的话，在论文里面其实意思是一致的，均可通用

> valse webinar: 
>
> - IL: 学了新东西不要忘了旧东西（Task- Domain- Class-）
> - LL: 增量学习过程有前向的传播，学了前面，对后面更好的（融会贯通）  和continual learning更像  更好地extend   从人的角度！ 
>
> 如果只是类别和任务什么的增加，可能是增量学习；但如果是数据增加，可能更偏向于终身学习



**continual learning vs. transfer learning**

从目的上来说，前者指在新旧的任务上都希望能够表现得很好，后者仅希望是应用前面的知识，使得当前任务表现得更好。



### Challenge

(放一张那个可塑性和xx性的图)

**Catastrophy forgetting ( main )**



**Intransigence**



### Method

#### Train from scratch

##### Data-Centric

回放式： 2个  直接+生成

数据惩罚：GEM+A-GEM

##### Model-Centric

Expansion(从backbone expansion方面着重讲把)

还有那个sota  TCIL 

##### Algorithm-Centric(感觉可以和上一个合并)

蒸馏+模型纠正   BiC 和 那个SSF有点相似的思想啊

**Discussion for from-scratch-method**

从数据集、哪种方法表现出强大实力（dynamic network）等方面来说

------

#### Pretrained

##### Prompt-Centric

L2P、Dual-Prompt、S-Prompt、CODA-Prompt

##### Parameter-efficient

ADAM (再看看还有没有其他的？)

##### multi-modal

+CLIP的

##### Other

ELLE(也是一种Expansion的方法)

**Discussion for pre-trained-method**

- 结合生物学思想
- transformer的decoder是否可以用来做generative的replay?
- 是否要结合rehearsal? 因为那篇综述里面说的，也可以加上rehearsal
- 在更具挑战性的benchmark上进行实验，而非那种from scratch的小实验