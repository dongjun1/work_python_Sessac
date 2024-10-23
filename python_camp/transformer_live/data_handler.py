'''
Data handling scripts.

1. class Vocabulary: Vocabulary wrapper class for generating/handling special tokens/index to word, and vice versa, etc....
'''

class Vocabulary:
    '''
    Handles vocabulary of a given squential datset.

    Arg:
        coverage(float): Coverage for determining whether the token shall be considered as OOV or not.

    Attributes:
        word2idx (dict[str, int]): Dict containing token as key, and index of the token as value.
        idx2word (dict[int, str]): Dict containing index of the token as key, and token as value.
        vocab_size (int): Integer representing the vocabulary size.

    Methods:

    Variables:
        SPECIAL_TOKENS
        EOS_IDX, EOS
        SOS_IDX, SOS
        PAD_IDX, PAD
        OOV_IDX, OOV
    '''
    EOS = '[EOS]'
    SOS = '[SOS]'
    PAD = '[PAD]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [EOS, SOS, PAD, OOV]

    def __init__(self, ):
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self.vocab_size: int = 0

        for special_token in Vocabulary.SPECIAL_TOKENS:
            self.add_word(special_token)
        
        self.eos_idx = self.word2idx[Vocabulary.EOS]
        self.sos_idx = self.word2idx[Vocabulary.SOS]
        self.pad_idx = self.word2idx[Vocabulary.PAD]
        self.oov_idx = self.word2idx[Vocabulary.OOV]

    def add_word(self, token: str) -> None:
        '''
        Adds a token to the vocabulary if it doesn't exists.
        If it exists, do nothing.

        Args:
            token (str): The token to be added
        '''
        if token not in self.word2idx:
            self.word2idx[token] = self.vocab_size
            self.idx2word[self.vocab_size] = token
            self.vocab_size += 1

    

