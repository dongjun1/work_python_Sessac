import torch
import torch.nn as nn
from data_handler import Vocabulary

            
def train_model(model, device, train_loader, criterion, source_vocab, target_vocab, 
                optimizer, num_epochs = 20, valid_loader = None, learning_rate = 0.001, print_ = False):
    
    model.to(device)
    model.train()
    loss = 0
    train_loss_history = []
    valid_loss_history = []
    optimizer = optimizer(model.parameters(), lr = learning_rate)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        correct, total = 0, 0
        for step_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch.to(device)
            target_batch.to(device)
            optimizer.zero_grad()
            
            pred_batch = model(source_batch, target_batch)
            batch_size, target_seq_length = target_batch.size()
            batch_size, source_seq_length = source_batch.size()
            loss = criterion(pred_batch.view(batch_size * target_seq_length, -1), target_batch.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss

            correct += torch.sum((torch.argmax(pred_batch, dim=2) == target_batch).float())
            total += target_batch.numel()

            if valid_loader is not None:
                valid_loss, valid_acc = eval_model(model, device, valid_loader, source_vocab, target_vocab, criterion, optimizer, print_ = print_)
                valid_loss_history.append(valid_loss)

        avg_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        train_total_acc = correct / total

        # Print the training progress
        if print_:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_total_acc * 100:.2f}%')
            if valid_loader is not None:
                print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc * 100:.2f}%')

        avg_loss = epoch_loss.item() / len(train_loader)
        train_loss_history.append(avg_loss)

    return train_loss_history, valid_loss_history

# 한국어 단어/ 영단어 몇개 골라서 오버피팅확인하기.
def eval_model(model, device, valid_loader, source_vocab, target_vocab, criterion, optimizer, num_epochs = 20, learning_rate = 0.001, print_ = False):
    model.to(device)
    model.eval()

    correct, total = 0, 0
    loss_list = []

    with torch.no_grad():
        for idx, (source_batch, target_batch) in enumerate(valid_loader):

            source_batch.to(device)
            target_batch.to(device)

            pred_batch = model(source_batch, target_batch).to(device)
            batch_size, target_seq_length = target_batch.size()
            batch_size, source_seq_length = source_batch.size()
            loss = criterion(pred_batch.view(batch_size * target_seq_length, -1), target_batch.view(-1))
            
            source_word_list, target_word_list, pred_word_list = \
            process_batches(source_batch = source_batch, target_batch = target_batch, 
                            pred_batch = pred_batch, source_vocab = source_vocab, target_vocab = target_vocab)
            
            if print_ and idx % 100 == 0:
                print_word(source_word_list, target_word_list, pred_word_list, source_seq_length = source_seq_length, 
                           target_seq_length = target_seq_length)

            loss_list.append(torch.mean(loss).item())
            # print(torch.argmax(pred_batch, dim = 2).shape)
            # print(target_batch.shape)
            # print(torch.argmax(pred_batch, dim = 2))
            # print(target_batch)
            correct += torch.sum((torch.argmax(pred_batch, dim = 2) == target_batch).float())
            total += target_batch.numel()
            
    return sum(loss_list) / len(loss_list), correct / total


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
