import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import defaultdict
from data_handler import LanguagePair, Vocabulary
from transformer_layers import Transformer
from typing import Callable, Dict, List, Optional, Tuple


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            criterion: nn.Module,
            optimizer: torch.optim.optimizer,
            save_dir: str = 'models/'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
    
    
    def train_model(
            self,
            train_data: DataLoader,
            valid_data: Optional[DataLoader] = None,
            num_epochs: int = 10,
            print_every: int = 1,
            evaluate_every: int = 1,
            evaluate_metrics: List[Callable] = [],
            source_vocab: Vocabulary = Vocabulary(),
            target_vocab: Vocabulary = Vocabulary()
    ) -> Tuple[List[float], List[float], Dict[str, List[float]], Dict[str, List[float]]]:
        train_loss_history: List[float] = []
        valid_loss_history: List[float] = []
        train_evaluation_result: defaultdict[List] = defaultdict(list)
        valid_evaluation_result: defaultdict[List] = defaultdict(list)
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_train_loss = 0
            epoch_train_eval: defaultdict[List] = defaultdict(list)

            for batch_idx, (src, tgt) in enumerate(train_data):
                output = self.model(src) # output logit
                pred_sent = self.get_token_from_logit(output, target_vocab) # list of tokens
                tgt_sent = self.get_token_from_logit(tgt, target_vocab)

                for metric in evaluate_metrics:
                    epoch_train_eval[metric.__name__].append(metric(tgt_sent, pred_sent))
                
                loss = self.criterion(tgt, output)
                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # == : call __eq__(), is : compare address
            if valid_data is not None and epoch % evaluate_every == 0:
                valid_loss, valid_metric, translate_result = self.evaluate(valid_data, evaluate_metrics, source_vocab, target_vocab)

            train_loss_history.append(epoch_train_loss)
            for metric in evaluate_metrics:
                t = epoch_train_eval[metric.__name__]
                train_evaluation_result[metric.__name__].append(sum(t) / len(t)) # 모든 batch_size가 동일하지 않을 수 있으므로 ex) 마지막 batch. 따라서 미약한 오류가 있음.
                valid_evaluation_result[metric.__name__].append(metric(tgt, pred))

        return train_loss_history, valid_loss_history, dict(train_evaluation_result), dict(valid_evaluation_result)

    def evaluate_model(
            self,
            valid_data: DataLoader,
            evaluate_metrics: List[Callable] = [],
            source_vocab: Vocabulary = Vocabulary(),
            target_vocab: Vocabulary = Vocabulary()
    ) -> Tuple[float, Dict[str, float], List[Tuple[List[str], List[str], List[str]]]]:
        
        self.model.eval()

        val_loss: int = 0
        evaluation_result: defaultdict[List] = defaultdict(list)
        translation: List = []

        for batch_idx, (src, tgt) in enumerate(valid_data):
            output = self.model(src) # output logit
            src_sent = self.get_tokens_from_indices(src, source_vocab)
            tgt_sent = self.get_tokens_from_indices(tgt, target_vocab)
            pred_sent = self.get_tokens_from_indices(output, target_vocab)

            for s, t, p in zip(src_sent, tgt_sent):
                translation.append((s, t, p))

            pred = self.get_token_from_logit(output, target_vocab) # list of tokens
            loss = self.criterion(tgt, output)
            val_loss += loss.item()

            for metric in evaluate_metrics:
                evaluation_result[metric.__name__].append(metric(output, pred))

            for metric, res in evaluation_result.items():
                evaluation_result[metric.__name__] = sum(res) / len(res)

        return val_loss, evaluation_result
            
        

    def get_token_from_logit(
            self,
            logit: torch.tensor,
            vocab: Vocabulary
    ) -> List[List[str]]:
        # logit.shape : batch_size, seq_length, vocab.vocab_size
        batch_size, seq_length, vocab_size = logit.size()
        assert vocab_size == vocab.vocab_size
        
        softmax = nn.Softmax(dim = -1)
        prob = softmax(logit)
        most_likely_tokens = torch.argmax(prob, dim = -1).tolist() # batch_size, seq_length

        

        return self.get_token_from_indices(most_likely_tokens, vocab)
    
    def get_tokens_from_indices(
            self,
            most_likely_tokens: List[List[int]],
            vocab: Vocabulary
    ) -> List[List[str]]:
        for sent_idx, sent in enumerate(most_likely_tokens):
            for tok_idx, token in enumerate(sent):
                i = most_likely_tokens[sent_idx][tok_idx]
                most_likely_tokens[sent_idx][tok_idx] = vocab.index2word(i)

    def get_logits(
            self,
            src: torch.tensor,
            tgt: Optional[torch.tensor] = None,
            target_vocab: Vocabulary = Vocabulary()
    ) -> torch.tensor:
        batch_size, seq_length, embedding_dim = src.size()
        decoder_input = torch.tensor([target_vocab.sos_idx for _ in range(batch_size)])

        while decoder_output == target_vocab.eos_idx:
            output = self.model(src, tgt)


if __name__ == '__main__':
    import config
    trainer = Trainer(device = config.device)


