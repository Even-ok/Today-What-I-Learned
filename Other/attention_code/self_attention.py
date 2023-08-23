import torch
import torch.nn.functional as F

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, heads):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model  # d_model = 128
        self.heads = heads  # 8个head
        self.d_head = d_model // heads
        
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        q = self.query(x)  # 投影到query空间
        k = self.key(x)  # 投影到key空间
        v = self.value(x)  # 投影到value空间
        
        q = q.view(batch_size, seq_len, self.heads, self.d_head)
        k = k.view(batch_size, seq_len, self.heads, self.d_head)     
        v = v.view(batch_size, seq_len, self.heads, self.d_head)    
        
        q = q.permute(0,2,1,3)  # 把seq_len和self.heads调一下位置
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        
        # 有scale， 然后矩阵相乘的话需要把k的后面两列换一下，即self.d_head和seq_len
        # 所以说和query的维度是保持一致的
        scores = torch.matmul(q, k.permute(0,1,3,2)) / (self.d_head**0.5)  
        # 所以最后是把d_head这一列给消掉了，仅仅保留seq_len这一项了！
        attention = F.softmax(scores, dim=-1) # 最后一维做attention，意思就是每个token，对整个序列所有token做attention，得到每个token和其它的相关值
        
        # bs, seq_len, heads, d_head
        out = torch.matmul(attention, v).permute(0,2,1,3).contiguous().view(
            batch_size, seq_len, self.d_model
        )  # 就是把head合并起来，返回原来的维度！ contiguous是深拷贝的意思
        
        return out
        
if __name__ == '__main__':
    input_data = torch.randn(4,10,128) # bs=4, seq=10, d_model=128
    attention_layer = SelfAttentionLayer(d_model=128, heads=8)
    output = attention_layer(input_data)
    print(output.shape)