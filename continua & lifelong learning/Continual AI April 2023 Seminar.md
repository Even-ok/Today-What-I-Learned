# Continual AI April 2023 Seminar

## Towards Continual Knowledge Learning ofLanguage Models ICLR'22

### Motivation

![image-20230828165438075](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828165438075.png)

知识在不断更新。但一些大语言模型，例如chatgpt，它被固定在2019年的知识库中了。

### Setting/Benchmark

![image-20230828171203334](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828171203334.png)

设置了4类任务：

- 第一个是一直以来有的旧知识，是不会改变的
- 第二个是随着时间不断更新的知识
- 第三个是包含factory resources，这些知识并不含在original pre-training dataset中的
- 第四个是第三个的简易版本

### Method

![image-20230828171502286](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828171502286.png)

对比了几种类别的CL方法，提到rehearsal是一种mix方式

### Experiment results

![image-20230828171645023](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828171645023.png)

- 提到rehearsal的表现最差，因为原始pre-train的数据量就已经很大了，而rehearsal不可能保留下所有的原始数据。用rehearsal的方法无异于在新的mix数据上进行预训练，效率不是很高，而且会hinder新数据的学习。这也说是原始continual learning和这种获取新世界知识的continual knowledge learning的区别！
- parameter-expansion的trade-off最好！

![image-20230828172327651](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828172327651.png)

- 记忆和遗忘是有相关性的，small-P1代表可以看到相同的数据8次（虽然我也不知道为啥），但这带来了更多的遗忘性。这个相同数据是指new data中的数据，而不是original数据，作者说它只看到了这些新数据这么多次，从而忘记了原始数据！

### Future work

![image-20230828172511772](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230828172511772.png)