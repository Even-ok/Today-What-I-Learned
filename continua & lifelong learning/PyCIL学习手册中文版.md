# PyCIL学习手册中文版

*Author: Even*



首先看一下main.py里面主要流程

```python
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)
```

实际上还是看那个param是怎么安排的，它赋值这个超参数的方式依据的是一个装载config的json文件。从最简单的"finetune.json"开始看一下共享的超参数是什么意思：

```json
{
    "prefix": "reproduce",   // 只是一个文件夹前缀命名而已
    "dataset": "cifar100",   // 采用的数据集
    "memory_size": 2000,     // 保留在memory buffer的总共样本量，k个类，每个类保存2000/k个样本
    "memory_per_class": 20,  // 固定每个类别放到memory buffer的最大数量
    "fixed_memory": false,   // 是否固定每个类别都放相同的数量
    "shuffle": true,
    "init_cls": 10,          // 初始阶段用于训练的类别数
    "increment": 10,         // 每个增量stage新增的class数量
    "model_name": "finetune",
    "convnet_type": "resnet32",
    "device": ["0","1","2","3"],
    "seed": [1993]           // 初始种子数
}
```



**注意**:exclamation:

1. 下载数据集的时候要关掉proxy，否则会报urlib的错。
2. 