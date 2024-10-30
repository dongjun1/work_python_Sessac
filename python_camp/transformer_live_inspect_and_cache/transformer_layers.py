import random
import torch
import torch.nn as nn
from typing import List, Optional
from data_handler import Vocabulary

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            max_length: int = 100
    ):
        super(PositionalEncoding, self).__init__()
        positional_encoding: torch.tensor = torch.zeros(max_length, embedding_dim)

        for p in range(position):
            for i in range(embedding_dim // 2):
                arg = (p / 10000) ** (2 * 1 / embedding_dim)
                positional_encoding[p][i] = torch.sin(arg)
                
                if 2 * 1 + 1 < embedding_dim:
                    positional_encoding[p][i+1] = torch.cos(arg)

        position: torch.tensor = torch.arange(0, max_length).float() # position.shape : (max_length, )
        i = torch.arange(0, embedding_dim, 2).float()
        arg = (position / 10000) ** (i / embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(arg)
        positional_encoding[:, 1::2] = torch.cos(arg)

        
        self.positional_encoding = positional_encoding

    def forward(
            self,
            x: torch.tensor
    ) -> torch.tensor:
    
        x = x + self.positional_encoding[:, :x.size(1)]
        return x
    

class TransformerDecoderLayer(nn.Module):
        def __init__(
                self,
                embedding_dim: int,
                num_heads: int,
                attention_head_dim: int,
                eps: float = 1e-6,
                max_length: int = 100
        ):
                super(TransformerDecoderLayer, self).__init__()
                self.masked_self_attention: MultiheadSelfAttention = MultiheadSelfAttention(embedding_dim, num_heads, attention_head_dim)
                self.enc_dec_self_attention: MultiheadSelfAttention = MultiheadSelfAttention(embedding_dim, num_heads, attention_head_dim)
                self.ff: nn.Linear = nn.Linear(embedding_dim, embedding_dim)
                self.layer_norm1: LayerNormalization = LayerNormalization(embedding_dim)
                self.layer_norm2: LayerNormalization = LayerNormalization(embedding_dim)
                self.layer_norm3: LayerNormalization = LayerNormalization(embedding_dim)
                
        def forward(
                self,
                x: torch.tensor,
                encoder_hidden: torch.tensor,
                t: int = 0
        ) -> torch.tensor:
                after_masked_self_atten = self.masked_self_attention(x, x, x, t)
                x = self.layer_norm1(x + after_masked_self_atten)
                
                enc_dec_self_atten = self.enc_dec_self_attention(x, encoder_hidden, encoder_hidden)
                x = self.layer_norm2(x + enc_dec_self_atten)
                
                after = self.ff(enc_dec_self_atten)
                x = self.layer_norm3(x + after)

                return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            attention_head_dim: int
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention: MultiheadSelfAttention = MultiheadSelfAttention(embedding_dim, num_heads, attention_head_dim)
        self.ff: nn.Linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm1: LayerNormalization = LayerNormalization(embedding_dim)
        self.layer_norm2: LayerNormalization = LayerNormalization(embedding_dim)
    
    def forward(
            self,
            x: torch.tensor
    ):
        after = self.self_attention(x, x, x)
        x = self.layer_norm1(x + after)
        
        after = self.ff(x)
        x = self.layer_norm2(x + after)

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embedding_dim: int,
            num_heads: int,
            attetion_head_dim: int
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers: List[TransformerEncoderLayer] = [TransformerEncoderLayer(embedding_dim, num_heads, attetion_head_dim) for _ in range(num_layers)]
        self.positional_encoding: PositionalEncoding = PositionalEncoding(embedding_dim)
    
    def forward(
            self,
            x: torch.tensor
    ) -> torch.tensor:
        
        x = x + self.positional_encoding()
        
        for layer in self.layers:
            x = layer(x)
        return x
    
class TransformerDecoder(nn.Module):
        def __init__(
                self,
                num_layers: int,
                embedding_dim: int,
                num_heads: int,
                attetion_head_dim: int,
        ):
                super(TransformerDecoder, self).__init__()

                self.layers: List[TransformerDecoderLayer] = [TransformerEncoderLayer(embedding_dim, num_heads, attetion_head_dim) for _ in range(num_layers)]
                self.positional_encoding: PositionalEncoding = PositionalEncoding(embedding_dim)

        def forward(
                self,
                x: torch.tensor,
                encoder_hidden: torch.tensor
        ) -> torch.tensor:
             # x.shape : batch_size, seq_length, embedding_dim
             x = x + self.positional_encoding()
             for layer in self.layers:
                x = layer(x, encoder_hidden)
             
             return x
    
class SelfAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            attetion_head_dim: int,
            masking_step: Optional[int, None] = None
    ):
        self.W_q: nn.Linear = nn.Linear(embedding_dim, attetion_head_dim)
        self.W_k: nn.Linear = nn.Linear(embedding_dim, attetion_head_dim)
        self.W_v: nn.Linear = nn.Linear(embedding_dim, attetion_head_dim)

        self.softmax: nn.Softmax = nn.Softmax(dim = -1)
        self.attention_head_dim = attetion_head_dim
        self.masking_step = masking_step
        
    def forward(
            self,
            query: torch.tensor,
            key: torch.tensor,
            value: torch.tensor,
            mask: Optional[int, None] = None
    ):
        Q: torch.tensor = self.W_q(query)
        K: torch.tensor = self.W_k(key)
        V: torch.tensor = self.W_v(value)

        score: torch.tensor = Q @ K.transpose(-2, -1) / self.attention_head_dim ** 0.5

        if not mask in None:
             mask = torch.tensor([1 for _ in range(self.masking_step)] + [eps for _ in range(x.size(1))])
             score = score * mask
        
        attention_distribution: torch.tensor = self.softmax(score)
        Z: torch.tensor = attention_distribution @ V

        return Z

class LayerNormalization(nn.Module):
    def __init__(
            self,
            input_dim: int,
            eps: float = 1e-6
    ):
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(
            self,
            x: torch.tensor
    ) -> torch.tensor:
        mean = torch.mean(x)
        std = torch.std(x)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            attetion_head_dim: int,
            batch_first: bool = True,
            masking_step: Optional[int, None] = None
    ):
        super(MultiheadSelfAttention, self).__init__()
        self.heads: List[SelfAttention] = [SelfAttention(embedding_dim, attetion_head_dim, masking_step) for _ in range(num_heads)]
        self.layer: nn.Linear = nn.Linear(num_heads * attetion_head_dim, embedding_dim)
        self.batch_first = batch_first
    
    def forward(
            self,
            x: torch.tensor
    ) -> torch.tensor:
        
        x = torch.cat([head(x) for head in self.heads], dim = 1)
        x = self.layer(x)

        return x
        
class Transformer(nn.Module):
    def __init__(
            self,
            num_heads: int,
            src_vocab: Vocabulary,
            tgt_vocab: Vocabulary,
            embedding_dim: int,
            num_layers: int,
            attetion_head_dim: int
    ):
        super(Transformer, self).__init__()
        src_vocab_size = src_vocab.vocab_size
        tgt_vocab_size = tgt_vocab.vocab_size

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder_embedding: nn.Embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.decoder_embedding: nn.Embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.encoder: TransformerEncoder = TransformerEncoder(num_layers, embedding_dim, num_heads, attetion_head_dim)
        self.decoder: TransformerDecoder = TransformerDecoder(num_layers, embedding_dim, num_heads)
        self.final_layer: nn.Linear = nn.Linear(embedding_dim, tgt_vocab_size)


    def forward(
            self,
            src: torch.tensor,
            tgt: torch.tensor,
            teacher_forcing_ratio: float = 0.5
    ) -> torch.tensor:
        assert src.size(0) == tgt.size(0)
        assert src.size(2) == tgt.size(2)

        batch_size, src_seq_length, embedding_dim = src.size()
        batch_size, tgt_seq_length, embedding_dim = tgt.size()
        src_embedding: torch.tensor = self.encoder_embedding(src)
        encoder_output: torch.tensor = self.encoder(src_embedding)
        
        tgt_embedding: torch.tensor = self.decoder_embedding(tgt) # tgt_embedding.shape : batch_size, seq_length, embedding_dim
        
        decoder_input: torch.tensor = torch.tensor([self.tgt_vocab.sos_idx for _ in range(batch_size)]) # decoder_input.shape : batch_size
        decoder_input = tgt_embedding(decoder_input).unsqueeze(1) # decoder_input.shape : batch_size, 1, embedding_dim

        output: torch.tensor = torch.full((batch_size, tgt_seq_length, self.tgt_vocab.vocab_size), self.tgt_vocab.pad_idx)
        t = 0
        eos_flag = torch.tensor([False for _ in range(batch_size)])
        # for t in range(tgt_embedding.size(1)):
        while True:
            decoder_output: torch.tensor = self.decoder(decoder_input, encoder_output)

            out = self.final_layer(decoder_output[:, -1])
            output[:, t] = out

            softmax = nn.Softmax(dim = -1)
            prob = softmax(out)
            y_t = torch.argmax(prob, dim = -1)

            if random() < teacher_forcing_ratio and t < tgt_embedding.size(1):
                decoder_input = torch.cat(decoder_input, tgt[:, t], dim = 1)
                
            else:
                decoder_input = torch.cat(decoder_input, tgt_embedding(y_t))
                eos_flag = eos_flag or (decoder_input[:, -1] == self.tgt_vocab.eos_idx)
            
            if all(eos_flag) or t > tgt_seq_length:
                break
            t += 1
        
        return out