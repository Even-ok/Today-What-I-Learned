import torch
import torch.nn.functional as F

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.ln1 = torch.nn.Linear(d_model, d_model * 2)
        self.ln2 = torch.nn.Linear(d_model * 2, d_model)
        self.dropout = dropout
        
        
    def forward(self, x):
        
        x = self.ln2(self.dropout(F.relu(self.ln1(x))))
        return x