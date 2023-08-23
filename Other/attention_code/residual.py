import torch
from multi_head_attention import MultiHeadAttention

class ResidualLayer(torch.nn.Module):
    def __init__(self, sublayer, d_model, dropout=0.1):
        super(ResidualLayer, self).__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.sublayer(self.norm(x)))
    # 先layernorm，再过一个layer，然后再过dropout，最后相加
    
if __name__ == '__main__':
    input_data = torch.randn(4,10,128) # bs=4, seq=10, d_model=128
    sublayer = MultiHeadAttention(d_model=128, heads=8)
    residual_layer = ResidualLayer(sublayer, d_model=128)
    output = residual_layer(input_data)
    print(output.shape)