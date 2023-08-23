import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)# [100, 1]
        div_term = torch.exp(torch.arange(0,
                    d_model,2,dtype=torch.float32) * (-math.log(10000.0)/d_model)) # 刚开始挺大的，然后就慢慢变小的权重，一维（d_model//2）
        pe[:, 0::2] = torch.sin(position * div_term) # 0::2的意思是说从0到最后，以2的step
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_seq_len, d_model]
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
if __name__ == '__main__':
    pos_encoding = PositionalEncoding(d_model=128, max_seq_len=100)
    input_data = torch.randn(4,100,128)
    output = pos_encoding(input_data)
    print(output.shape)