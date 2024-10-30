import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BatchNormalization(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = 1e-6
        self.batch_dim = 0

    def forward(self, x):
        mean = x.mean(dim = self.batch_dim)
        std = x.var(dim = self.batch_dim)
        x_hat = (x - mean) / torch.sqrt(std + self.eps)

        return self.gamma * x_hat + self.beta
    
class DropOut(nn.Module):
    def __init__(self, dropout_rate):
        super(DropOut, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        dropout_mask = (torch.rand(x.shape) > self.dropout_rate).float()
        x = x * dropout_mask
        x /= (1 - self.dropout_rate)

        return x
    
class L2Regularization(nn.Module):
    def __init__(self,):
        pass
