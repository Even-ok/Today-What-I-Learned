### PET各个模块学习



#### adapter

```python
        (adapters): ModuleDict(
          (attn): Adapter(
            (layer): Sequential(
              (0): Linear(in_features=768, out_features=5, bias=True)
              (1): GELU(approximate=none)
              (2): Linear(in_features=5, out_features=768, bias=True)
              (3): Scaler(scale=1.0000, learnable=True)
            )
          )
        )
```

其实就是在attention后输出的token后面（MLP后）再插入一个线性层



#### LoRA

```python
    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, module: nn.Linear, input: torch.Tensor):
        weight = self.lora_B @ self.lora_A
        output = F.linear(input, module.weight + weight, module.bias)  # 原来weight是直接加在module.weight上的
        return output
```

