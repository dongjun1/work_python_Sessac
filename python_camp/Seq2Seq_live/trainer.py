import torch
import torch.nn as nn
from data_handler import Vocabulary

def train_model(model, train_loader, criterion, source_vocab, target_vocab, optimizer, num_epochs = 20, valid_loader = None, learning_rate = 0.001):
    model.train()
    loss = 0
    train_loss_history = []
    valid_loss_history = []
    optimizer = optimizer(model.parameters(), lr = learning_rate)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        for step_idx, (source_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            pred_batch = model(source_batch, target_batch)
            batch_size, seq_length = target_batch.size()
            loss = criterion(pred_batch.view(batch_size * seq_length, -1), target_batch.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss

            valid_loss, valid_acc = eval_model(model = model, valid_loader = valid_loader, 
                                               source_vocab = source_vocab, target_vocab = target_vocab, 
                                               criterion = criterion, optimizer = optimizer, num_epochs = num_epochs, learning_rate = learning_rate)
            valid_loss_history.append(valid_loss)

            if step_idx % 100 == 0:
                # print(f'[Epoch {epoch} / {num_epochs}], step {step_idx} : train loss - {loss}')
                print(f'[Epoch {epoch} / {num_epochs}], step {step_idx} : train loss - {loss}, valid loss - {valid_loss}, valid_acc - {valid_acc * 100:.2f}%')
                # print(f'[Epoch {epoch} / {num_epochs}], step {step_idx} : train loss - {loss}, valid loss - {valid_loss}')
        
        avg_loss = epoch_loss.item() / len(train_loader)
        train_loss_history.append(avg_loss)

    return train_loss_history, valid_loss_history

# 한국어 단어/ 영단어 몇개 골라서 오버피팅확인하기.
def eval_model(model, valid_loader, source_vocab, target_vocab, criterion, optimizer, num_epochs = 20, learning_rate = 0.001):
    model.eval()

    correct, total = 0, 0
    loss_list = []

    with torch.no_grad():
        for idx, (source_batch, target_batch) in enumerate(valid_loader):

            pred_batch = model(source_batch, target_batch)
            batch_size, seq_length = target_batch.size()
            batch_size, source_seq_length = source_batch.size()
            loss = criterion(pred_batch.view(batch_size * seq_length, -1), target_batch.view(-1))

            pred_word_list = []
            source_word_list = []
            target_word_list = []
            for pred_seq in torch.argmax(pred_batch, dim = 2):
                for pred_word in pred_seq:
                    # print(f'pred tensor : {pred_word}')
                    word = target_vocab.index_to_word(pred_word.item())
                    # print(f'pred word : {word}')
                    pred_word_list.append(word)

            for source_seq in source_batch:
                for source_word in source_seq:
                    # print(f'source tensor : {source_word}')
                    word = source_vocab.index_to_word(source_word.item())
                    # print(f'source word : {word}') 
                    source_word_list.append(word)

            for target_seq in target_batch:
                for target_word in target_seq:
                    # print(f'source tensor : {source_word}')
                    word = target_vocab.index_to_word(target_word.item())
                    # print(f'source word : {word}') 
                    target_word_list.append(word)
            
            for sent_idx in range(len(source_word_list) // source_seq_length):
                print('Source: ' + '\t'.join(source_word_list[sent_idx*source_seq_length:(sent_idx+1)*source_seq_length]))
                print('Target: ' + '\t'.join(target_word_list[sent_idx*seq_length:(sent_idx+1)*seq_length]))
                print('Prediction: ' + '\t'.join(pred_word_list[sent_idx*seq_length:(sent_idx+1)*seq_length]))
            
            if idx > 2:
                return 

            loss_list.append(torch.mean(loss).item())
            # print(torch.argmax(pred_batch, dim = 2).shape)
            # print(target_batch.shape)
            # print(torch.argmax(pred_batch, dim = 2))
            # print(target_batch)
            correct += torch.sum((torch.argmax(pred_batch, dim = 2) == target_batch).float())
            total += target_batch.numel()
            
    return sum(loss_list) / len(loss_list), correct / total
            