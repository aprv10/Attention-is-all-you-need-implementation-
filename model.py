import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    #d_model -> dim of model
    def __init__(self, d_model: int, vocab_size: int):  
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #sqrt(d) given in paper

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_length:int, droput: float) -> None:    
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(droput)

        #matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_length, d_model)
        #vector of shape(seq_len, 1)
        #positional enc in log space(slightly modified formulae)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #sin to even pos and cos to odd
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) #(1,seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplicative term
        self.bias = nn.Parameter(torch.zeros(1)) #additive term

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha*(x-mean)/(std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 & B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 & B2

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))