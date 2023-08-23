import torch
import torch.nn.functional as F

class CrossMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(CrossMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x, enc_output):
        batch_size, seq_len, d_model = x.size()
        
        q = self.wq(x)
        k = self.wk(enc_output)
        v = self.wv(enc_output)
        
        q = q.view(batch_size, seq_len, self.heads, self.d_head)
        k = k.view(batch_size, seq_len, self.heads, self.d_head)     
        v = v.view(batch_size, seq_len, self.heads, self.d_head)   
        
        q = q.permute(0,2,1,3)  # 把seq_len和self.heads调一下位置
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        
        scores = torch.matmul(q,k.permute(0,1,3,2)) / (self.d_head) ** 0.5
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(self.dropout(attention), v).permute(0,2,1,3).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.fc(out)
    
