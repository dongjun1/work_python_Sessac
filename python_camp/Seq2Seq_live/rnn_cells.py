import torch
import torch.nn as nn

# RNNCell's input = step, RNNBuiltIn's input = sequence
# this step, use RNNCell, because use encoder, decoder both
class RNNCellManual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCellManual, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x_t, h_t):
        '''
        Args:
            x_t : batch_size, input_dim
            h_t : hidden_dim
        Return:
            batch_size, hidden_dim
        '''
        batch_size = x_t.size(0)
        assert x_t.size(1) == self.input_dim, f'Input dimension was expected to be {self.input_dim}, got {x_t.size(1)}'
        assert h_t.size(0) == batch_size, f'0th dimension of output of RNNManualCell is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension of output of RNNManualCell is expected to be {self.hidden_dim}, got {h_t.size(1)}'
        
        h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))

        assert h_t.size(0) == batch_size, f'0th dimension of output of RNNManualCell is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension of output of RNNManualCell is expected to be {self.hidden_dim}, got {h_t.size(1)}'

        return h_t
    
    def initialize(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

class LSTMCellManual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCellManual, self).__init__()
        self.i2i = nn.Linear(input_dim, hidden_dim)
        self.h2i = nn.Linear(hidden_dim, hidden_dim)
        self.i2f = nn.Linear(input_dim, hidden_dim)
        self.h2f = nn.Linear(hidden_dim, hidden_dim)
        self.i2g = nn.Linear(input_dim, hidden_dim)
        self.h2g = nn.Linear(hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x_t, h_t, c_t):
        batch_size = x_t.size(0)
        assert x_t.size(1) == self.input_dim
        
        assert h_t.size(0) == batch_size
        assert h_t.size(1) == self.hidden_dim

        assert c_t.size(0) == batch_size
        assert c_t.size(1) == self.hidden_dim

        i_t = torch.sigmoid(self.i2i(x_t) + self.h2i(h_t))
        f_t = torch.sigmoid(self.i2f(x_t) + self.h2f(h_t))
        g_t = torch.tanh(self.i2g(x_t) + self.h2g(h_t))
        o_t = torch.sigmoid(self.i2o(x_t) + self.h2o(h_t))

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
    def initialize(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim), torch.zeros(batch_size, self.hidden_dim)