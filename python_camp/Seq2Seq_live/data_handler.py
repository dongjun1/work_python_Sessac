import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from collections import defaultdict
from debug_shell import debug_shell

class Vocabulary:
    PAD = '[PAD]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [PAD, SOS, EOS, OOV]
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    OOV_IDX = 3

    def __init__(self, word_count, coverage = 0.999):
        '''
        Accept word_count dictionary having word as key, and frequency as value.
        '''
        word_freq_list = []
        total = 0
        for word, freq in word_count.items():
            word_freq_list.append((word, freq))
            total += freq

        word_freq_list = sorted(word_freq_list, key = lambda x : x[1], reverse = True) # freq
        word_list = []
        word2index = {}
        index2word = {}
        s = 0

        for idx, (word, freq) in enumerate([(e, 0) for e in Vocabulary.SPECIAL_TOKENS] + word_freq_list):
            s += freq
            if s > coverage * total:
                break
            word2index[word] = idx
            index2word[idx] = word

        self.word2idx = word2index
        self.idx2word = index2word
        self.vocab_size = len(word2index)

    def word_to_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        
        return self.OOV_IDX



def parse_file(file_path, train_valid_test_ratio = (0.8, 0.1, 0.1), batch_size = 32):
    f = open(file_path, 'r', encoding = 'utf-8')
    data = []
    source_word_count = defaultdict(int) # == defaultdict(lambda : 0)
    target_word_count = defaultdict(int)

    for line in f.readlines():
        line = line.strip()
        source, target, etc = line.split('\t')

        source = source.split()

        for source_token in source:
            source_word_count[source_token] += 1
        
        target = target.split()

        for target_token in target:
            target_word_count[target_token] += 1

        data.append((source, target))

    source_vocab = Vocabulary(source_word_count)
    target_vocab = Vocabulary(target_word_count)

    for idx, (source, target) in enumerate(data):
        data[idx] = (list(map(source_vocab.word_to_index, source)), list(map(target_vocab.word_to_index, target)))

    lengths = [int(len(data) * ratio) for ratio in train_valid_test_ratio]
    lengths[-1] = len(data) - sum(lengths[:-1])
    datasets = random_split(data, lengths)
    dataloaders = [DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: preprocessing(x, source_vocab, target_vocab)) for dataset in datasets]

    return dataloaders, source_vocab, target_vocab

def preprocessing(batch, source_vocab, target_vocab):
    sources = [e[0] for e in batch]
    targets = [e[1] for e in batch]

    source_seqs = []
    target_seqs = []

    for source_seq in sources:
        source_seqs.append(source_seq + [source_vocab.EOS_IDX])

    for target_seq in targets:
        target_seqs.append([target_vocab.SOS_IDX] + target_seq + [target_vocab.EOS_IDX])

    source_max_length = max([len(s) for s in source_seqs])
    target_max_length = max([len(s) for s in target_seqs])

    for idx, seq in enumerate(source_seqs):
        seq = seq + [source_vocab.PAD_IDX] * (source_max_length - len(seq))
        assert len(seq) == source_max_length, f'Expected to have {source_max_length}, now {len(seq)}'
        source_seqs[idx] = seq

    for idx, seq in enumerate(target_seqs):
        seq = seq + [target_vocab.PAD_IDX] * (target_max_length - len(seq))
        assert len(seq) == target_max_length, f'Expected to have {target_max_length}, now {len(seq)}'
        target_seqs[idx] = seq

    return torch.tensor(source_seqs), torch.tensor(target_seqs)

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
            h_t : batch_size, hidden_dim
        Return:
            batch_size, hidden_dim
        '''
        batch_size = x_t.size(0)
        assert x_t.size(1) == self.input_dim, f'Input dimension was expected to be {self.input_dim}, got {x_t.size(1)}'
        assert h_t.size(0) == batch_size, f'0th dimension was expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension was expected to be {self.hidden_dim}, got {h_t.size(1)}'
        h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))

        assert h_t.size(0) == batch_size, f'0th dimension of output of RNNManualCell is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension of output of RNNManualCell is expected to be {self.hidden_dim}, got {h_t.size(1)}'

        return h_t
    
    def initialize(self):
        return torch.zeros(self.hidden_dim)

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
    
    def initialize(self):
        return torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)
    
class EncoderState:
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')

    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()
    
class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)

    def forward(self, source):
        # example = [[1, 3, 2, 1, 1], [2, 3, 1, 3, 1]]
        # batch_size : 2 / seq : 5, source_vocab : 3
        batch_size, seq = source.size()
        
        # embedded.shape = batch_size, seq, embedding_dim
        # e : nn.Embedding
        # e(1) = [1.5, 1.2], e(2) = [2.5, -0.7], e(3) = [-1.7, 0.2]
        # self.embedding(example) = [[[1.5, 1.2], [-1.7, 0.2], [2.5, -0.7], [1.5, 1.2], [1.5, 1.2]], [[2.5, -0.7], [-1.7, 0.2], [1.5, 1.2], [-1.7, 0.2], [1.5, 1.2]]]
        embedded = self.embedding(source)
        encoder_state = EncoderState(model_type = self.model_type).initialize()

        for t in range(seq):
            x_t = embedded[:, t, :]
            
            encoder_state = self.cell(x_t, *encoder_state)

        return encoder_state

class Seq2Seq(nn.Moudle):
    def __init___(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):
        encoder_hidden = self.encoder(source)
        outputs = self.decoder(target, encoder_hidden)

        return outputs
    



if __name__ == '__main__':
    batch_size = 32
    (train, valid, test), source_vocab, target_vocab = parse_file('kor-eng/kor.txt', batch_size = batch_size)
    
    encoder = Encoder(source_vocab = source_vocab, embedding_dim = 8, hidden_dim = 32, model_type = RNNCellManual)
    
    for source_batch, target_batch in train:
        assert source_batch.shape[0] == batch_size
        print(source_batch)

        assert target_batch.shape[0] == batch_size
        print(target_batch)
        # print(source_batch.shape[0])
        # print(target_batch.shape[0])