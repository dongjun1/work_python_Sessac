from collections import defaultdict
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

class Vocabulary:
    PAD = '[PAD]'
    OOV = '[OOV]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    PAD_IDX = 0
    OOV_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIAL_TOKENS = [PAD, OOV, SOS, EOS]

    def __init__(self, word_count, coverage = 0.999):
        word_freq_list = []
        total = 0
        for word, freq in word_count.items():
            word_freq_list.append((word, freq))
            total += freq

        word_freq_list = sorted(word_freq_list, key = lambda x : x[1], reverse = True) # sort by freq descending

        word_to_idx = {}
        idx_to_word = {}
        t = 0

        for idx, (word, freq) in enumerate([(e, 0) for e in self.SPECIAL_TOKENS] + word_freq_list):
            t += freq
            if t > coverage * total:
                break
        
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = len(word_to_idx)

    def get_word_to_idx(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]

        return self.OOV_IDX

def get_texts(file_path, train_valid_ratio = (0.8, 0.1, 0.1), batch_size = 32):
    f = open(file_path, 'r', encoding = 'utf-8')

    # want calculate each word's frquency from eng_senctence, kor_sentence
    eng_word_count = defaultdict(lambda : 0) # == defaultdict(int)
    kor_word_count = defaultdict(lambda : 0)
    data = []

    # think more how to split symbol
    for line in f.readlines():
        line = line.strip()
        
        eng_sentence, kor_sentence, _ = line.split('\t')
        
        eng_sentence = eng_sentence.split()

        for eng in eng_sentence:
            eng_word_count[eng] += 1

        kor_sentence = kor_sentence.split()

        for kor in kor_sentence:
            kor_word_count[kor] += 1
        
        data.append((eng_sentence, kor_sentence))
    
    eng_vocab = Vocabulary(eng_word_count)
    kor_vocab = Vocabulary(kor_word_count)

    # make index vector, each sentences
    for idx, (eng_sentence, kor_sentence) in enumerate(data):
        data[idx] = (list(map(eng_vocab.get_word_to_idx, eng_sentence)), list(map(kor_vocab.get_word_to_idx, kor_sentence))) # return ([eng_word's index], [kor_word's index])

    lenghts = [int(len(data) * ratio) for ratio in train_valid_ratio]
    lenghts[-1] = len(data) - sum(lenghts[:-1]) # Calibrate in case the number of data in the testing set is incorrect
    datasets = random_split(data, lenghts)
    dataloaders = [DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: preprocessing(x, eng_vocab, kor_vocab)) for dataset in datasets]

    return dataloaders, eng_vocab, kor_vocab

def preprocessing(batch_dataset, eng_vocab, kor_vocab):
    eng_data = [e[0] for e in batch_dataset]
    kor_data = [e[1] for e in batch_dataset]

    eng_seq = []
    kor_seq = []

    for eng in eng_data:
        eng_seq.append(eng + [eng_vocab.EOS_IDX])
    
    for kor in kor_data:
        kor_seq.append([kor_vocab.SOS_IDX] + kor + [kor_vocab.EOS_IDX])

    eng_max_length = max([len(s) for s in eng_seq])
    kor_max_length = max([len(s) for s in kor_seq])

    for idx, seq in enumerate(eng_seq):
        seq = seq + [eng_vocab.PAD_IDX] * (eng_max_length - len(seq))
        assert len(seq) == eng_max_length, f'Expected {eng_max_length}, now {len(seq)}'
        eng_seq[idx] = seq

    for idx, seq in enumerate(kor_seq):
        seq = seq + [kor_vocab.PAD_IDX] * (kor_max_length - len(seq))
        assert len(seq) == kor_max_length, f'Expected {kor_max_length}, now {len(seq)}'
        kor_seq[idx] = seq
    
    return torch.tensor(eng_seq), torch.tensor(kor_seq)


    

if __name__ == '__main__':
    path = 'kor-eng/kor.txt'

    (train, valid, test), eng_vocab, kor_vocab = get_texts(path)

    

    

    


    