import random
import torch
import torch.nn as nn
from attentions import LuongAttention, BahdanauAttention
from rnn_cells import RNNCellManual, LSTMCellManual

# add to.(device)
# if make the instance of class.to(device) they have attribute is go to device, but make new tensor is not gone, so using to.(device)

class EncoderState:
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')

    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()
    
class Encoder(nn.Module):
    def __init__(self, device, source_vocab, embedding_dim, hidden_dim, model_type):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.device = device

        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim).to(self.device)
        self.cell = model_type(embedding_dim, hidden_dim, self.device).to(self.device)

    def forward(self, source):
        # example = [[1, 3, 2, 1, 1], [2, 3, 1, 3, 1]]
        # batch_size : 2 / seq : 5, source_vocab : 3
        batch_size, seq = source.size()
        hiddens = []

        source = source.to(self.device)
        
        # embedded.shape = batch_size, seq, embedding_dim
        # e : nn.Embedding
        # e(1) = [1.5, 1.2], e(2) = [2.5, -0.7], e(3) = [-1.7, 0.2]
        # self.embedding(example) = [[[1.5, 1.2], [-1.7, 0.2], [2.5, -0.7], [1.5, 1.2], [1.5, 1.2]], [[2.5, -0.7], [-1.7, 0.2], [1.5, 1.2], [-1.7, 0.2], [1.5, 1.2]]]
        embedded = self.embedding(source).to(self.device)
        encoder_state = self.cell.initialize(batch_size).to(self.device)

        for t in range(seq):
            x_t = embedded[:, t, :].to(self.device)
            if self.model_type == RNNCellManual:
                encoder_state = self.cell(x_t, encoder_state).to(self.device)
                hiddens.append(encoder_state)
            elif self.model_type == LSTMCellManual:
                encoder_state = self.cell(x_t, *encoder_state)
                hiddens.append(encoder_state[0].to(self.device))

        return torch.stack(hiddens, dim = 1).to(self.device)
    
class Decoder(nn.Module):
    def __init__(self, device, target_vocab, embbeding_dim, hidden_dim, model_type, attention):
        super(Decoder, self).__init__()
        self.target_vocab = target_vocab
        self.embbeding_dim = embbeding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.device = device
        
        if attention == 'LuongAttention':
            self.attention = LuongAttention(self.device)
            self.W_c = nn.Linear(target_vocab.vocab_size + hidden_dim, target_vocab.vocab_size).to(device)
            self.embedding = nn.Embedding(target_vocab.vocab_size, embbeding_dim).to(self.device)
            self.cell = model_type(embbeding_dim, hidden_dim, self.device).to(self.device)
        elif attention == 'BahdanauAttention':
            self.attention = BahdanauAttention(self.device)
            self.cell = model_type(embbeding_dim + hidden_dim, hidden_dim, self.device).to(self.device)
        else:
            self.embedding = nn.Embedding(target_vocab.vocab_size, embbeding_dim).to(self.device)
            self.cell = model_type(embbeding_dim, hidden_dim, self.device).to(self.device)
            
        self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size).to(self.device)
        
        

    def forward(self, target, encoder_hiddens, teacher_forcing_ratio = 0.5):
        # target.shape : batch_size, seq_length
        batch_size, seq_length = target.size()
        
        encoder_last_state = encoder_hiddens[:, -1].to(self.device)
        outputs = []
        input = torch.tensor([self.target_vocab.SOS_IDX for _ in range(batch_size)]).to(self.device) # input : batch_size

        decoder_state = encoder_last_state

        for t in range(seq_length):
            # embbeded.shape : batch_size, embbeding_dim
            embedded = self.embedding(input).to(self.device)
            if isinstance(self.attention, BahdanauAttention):
                context_vector = self.attention(encoder_hiddens, decoder_state).to(self.device)
                embedded = torch.cat((embedded, context_vector), dim = 1).to(self.device)
                # embedded.shape : batch_size, hidden_dim + embedding_dim
            
            if self.model_type == RNNCellManual:
                decoder_state = self.cell(embedded, decoder_state).to(self.device)
            elif self.model_type == LSTMCellManual:
                decoder_state = self.cell(embedded, *decoder_state)
            output = self.h2o(decoder_state).to(self.device)
            
            if isinstance(self.attention, LuongAttention):
                context_vector = self.attention(decoder_state, encoder_hiddens).to(self.device)
                output = torch.cat((output, context_vector), dim = 1).to(self.device)
                # output.shape : batch_size, (self.output_dim + encoder_hidden_dim)
                output = torch.tanh(self.W_c(output)).to(self.device)
                # output.shape : batch_size, target_vocab.vocab_size
            outputs.append(output)

            # random.random() : return 0 ~ 1
            if random.random() < teacher_forcing_ratio and t < seq_length - 1: # do teahcer forcing
                input = target[:, t + 1].to(self.device)
            else:
                input = torch.argmax(output, dim = 1).to(self.device)

        # outputs.shape : seq_length, batch_size, vocab_size
        return torch.stack(outputs, dim = 1).to(self.device) # outputs.shape : batch_size, seq_length, vocab_size


class Seq2Seq(nn.Module):
    def __init__(self, device, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target):
        source = source.to(self.device)
        target = target.to(self.device)
        encoder_hiddens = self.encoder(source).to(self.device)
        outputs = self.decoder(target, encoder_hiddens).to(self.device)

        return outputs
    

