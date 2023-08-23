import torch
from multi_head_attention import MultiHeadAttention
from residual import ResidualLayer
from feed_forward import FeedForwardLayer
from cross_multi_head_attention import CrossMultiHeadAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, heads, dropout)
        self.residual1 = ResidualLayer(self.multihead_attention, d_model, dropout)
        self.feed_forward = FeedForwardLayer(d_model, dropout)
        self.residual2 = ResidualLayer(self.feed_forward, d_model, dropout)
        
    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attention = MultiHeadAttention(d_model, heads, dropout) # 这个应该是self的？
        self.residual1 = ResidualLayer(self.masked_multihead_attention, d_model, dropout)
        self.cross_multihead_attention = CrossMultiHeadAttention(d_model, heads, dropout)
        self.residual2 = ResidualLayer(self.cross_multihead_attention, d_model, dropout)
        self.feed_forward = FeedForwardLayer(d_model, dropout)
        self.residual3 = ResidualLayer(self.feed_forward, d_model, dropout)
        
    def forward(self, x, enc_output):
        x = self.residual1(x)  # 放进来的是query\
        x = x + self.dropout(self.cross_multihead_attention(self.norm(x), enc_output))
        x = self.residual3(x)
        
        
        
        
        