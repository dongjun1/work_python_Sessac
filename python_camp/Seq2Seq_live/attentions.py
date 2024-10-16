import torch
import torch.nn as nn

class LuongAttention(nn.Module):
    def __init__(self, device):
        super(LuongAttention, self).__init__()
        self.device = device
    
    def forward(self, decoder_state, encoder_hiddens):
        batch_size, encoder_seq_length, hidden_dim = encoder_hiddens.size()
        attention_score = torch.zeros(batch_size, encoder_seq_length).to(self.device)
        s_t = decoder_state

        for t in range(encoder_seq_length):
            h_t = encoder_hiddens[:, t].to(self.device)
            attention_score[:, t] = torch.sum(s_t * h_t).to(self.device)

        attention_distribution = torch.softmax(attention_score, dim = 1).to(self.device)

        context_vector = torch.zeros(batch_size, hidden_dim).to(self.device)
        
        for t in range(encoder_seq_length):
            context_vector += attention_distribution[:, t].unsqueeze(1) * encoder_hiddens[:, t]

        return context_vector

class BahdanauAttention(nn.Module):
    def __init__(self, k, h, device):
        # k : hidden_dim for attention
        super(BahdanauAttention, self).__init__()
        self.device = device
        self.W_a = nn.Linear(k, 1)
        self.W_b = nn.Linear(h, k) #s_t-1 = h, 1 -> k, 1
        self.W_c = nn.Linear(h, k) # W_c * H, H.shape : h, L -> k, L

    def forward(self, decoder_state, encoder_hiddens):
        attention_score = self.W_a(torch.tanh(self.W_b(decoder_state) + self.W_c(encoder_hiddens))).to(self.device)
        attention_distribution = torch.softmax(attention_score, dim = 1).to(self.device)

        batch_size, encoder_seq_length, hidden_dim = encoder_hiddens.size()

        attention_distribution = nn.Softmax(attention_score, dim = 1).to(self.device)

        context_vector = torch.zeros(batch_size, hidden_dim).to(self.device)
        
        for t in range(encoder_seq_length):
            context_vector += attention_distribution[:, t] * encoder_hiddens[:, t]

        return context_vector



