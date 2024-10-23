import torch
import torch.nn as nn
from data_handler import Vocabulary
from metrics import *


            
def train_model(model, device, train_loader, criterion, source_vocab, target_vocab, 
                optimizer, num_epochs = 20, valid_loader = None, learning_rate = 0.001, print_ = False):
    
    model.to(device)
    model.train()
    loss = 0
    train_losses = []
    valid_losses = []
    valid_metrics = []
    optimizer = optimizer(model.parameters(), lr = learning_rate)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        correct, total = 0, 0
        for step_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch.to(device)
            target_batch.to(device)
            optimizer.zero_grad()
            # print(f'source : {source_batch}')
            # print(f'target : {target_batch}')
            # print(f'source_shape : {source_batch.shape}')
            # print(f'target_shape : {target_batch.shape}')
            pred_batch = model(source_batch, target_batch).to(device)
            batch_size, target_seq_length = target_batch.size()
            batch_size, source_seq_length = source_batch.size()
            loss = criterion(pred_batch.view(batch_size * target_seq_length, -1), target_batch.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            correct += torch.sum((torch.argmax(pred_batch, dim=2) == target_batch).float())
            total += target_batch.numel()

            if valid_loader is not None:
                valid_loss, valid_score = eval_model(model, device, valid_loader, source_vocab, target_vocab, criterion, optimizer)
                valid_losses.append(valid_loss)
                valid_metrics.append(valid_score)
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        train_total_acc = correct / total

        # Print the training progress
        if print_:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_total_acc * 100:.2f}%')
            if valid_loader is not None:
                print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc * 100:.2f}%')

    return train_losses, valid_losses, valid_metrics

# 한국어 단어/ 영단어 몇개 골라서 오버피팅확인하기. ok
# hyperparameter를 자동으로 찾아가게끔 코드 작성.
# ex. 학습 중 loss가 떨어지는 폭이 작으면 학습을 중단하고 다음 hyperparameter로 다시 처음부터 학습을 진행하도록...
def eval_model(model, device, valid_loader, source_vocab, target_vocab, criterion, optimizer, num_epochs = 20, learning_rate = 0.001, print_ = False, evaluate_metric = 'accuracy'):
    model.to(device)
    model.eval()

    
    eval_loss = 0

    if evaluate_metric == 'accuracy':
        correct, total = 0, 0
    elif evaluate_metric == 'BLEU':
        bleu_sum = 0

    with torch.no_grad():
        for idx, (source_batch, target_batch) in enumerate(valid_loader):

            source_batch.to(device)
            target_batch.to(device)

            pred_batch = model(source_batch, target_batch).to(device)
            batch_size, target_seq_length = target_batch.size()
            batch_size, source_seq_length = source_batch.size()
            
            
            source_word_list, target_word_list, pred_word_list = \
            process_batches(source_batch = source_batch, target_batch = target_batch, 
                            pred_batch = pred_batch, source_vocab = source_vocab, target_vocab = target_vocab)
            
            if print_ and idx % 100 == 0:
                print_word(source_word_list, target_word_list, pred_word_list, source_seq_length = source_seq_length, 
                           target_seq_length = target_seq_length)

            if evaluate_metric == 'accuracy':
                pred = torch.argmax(pred_batch, dim = 2)
                correct += torch.sum((pred == target_batch).float())
                total += target_batch.numel()
            elif evaluate_metric == 'BLEU':
                pred = torch.argmax(pred_batch, dim = 2)
                bleu_sum = sum([bleu(pred_sent, target_sent) for pred_sent, target_sent in zip(pred, target_batch)] / len(target_batch))

            loss = criterion(pred_batch.view(batch_size * target_seq_length, -1), target_batch.view(-1))
            eval_loss += loss.item()

            if evaluate_metric == 'accuracy':
                metric = correct / total
            elif evaluate_metric == 'BLEU':
                metric = bleu_sum / len(valid_loader)
            
            avg_val_loss = eval_loss / len(valid_loader)
            print(f'Validation Loss : {avg_val_loss:.4f}, Validation {evaluate_metric} : {metric:.4f}')
            
    return avg_val_loss, metric


def process_batches(source_batch, target_batch, pred_batch, source_vocab, target_vocab):
    # Initialize lists to store words
    pred_word_list = []
    source_word_list = []
    target_word_list = []
    
    # Get the batch and sequence lengths
    batch_size, seq_length = target_batch.size()
    batch_size, source_seq_length = source_batch.size()
    
    # Convert predicted tensor sequences to word lists
    for pred_seq in torch.argmax(pred_batch, dim=2):
        for pred_word in pred_seq:
            word = target_vocab.index_to_word(pred_word.item())
            pred_word_list.append(word)

    # Convert source tensor sequences to word lists
    for source_seq in source_batch:
        for source_word in source_seq:
            word = source_vocab.index_to_word(source_word.item())
            source_word_list.append(word)

    # Convert target tensor sequences to word lists
    for target_seq in target_batch:
        for target_word in target_seq:
            word = target_vocab.index_to_word(target_word.item())
            target_word_list.append(word)
    
    return source_word_list, target_word_list, pred_word_list
    

def print_word(*word_list, source_seq_length, target_seq_length):
    source_word_list, target_word_list, pred_word_list = word_list

    # Print out the source, target, and predicted sequences for each sentence in the batch
    for sent_idx in range(len(source_word_list) // source_seq_length):
        source_sentence = '\t'.join(source_word_list[sent_idx*source_seq_length:(sent_idx+1)*source_seq_length])
        target_sentence = '\t'.join(target_word_list[sent_idx*target_seq_length:(sent_idx+1)*target_seq_length])
        pred_sentence = '\t'.join(pred_word_list[sent_idx*target_seq_length:(sent_idx+1)*target_seq_length])
        
        print(f'Source: {source_sentence}')
        print(f'Target: {target_sentence}')
        print(f'Prediction: {pred_sentence}')
