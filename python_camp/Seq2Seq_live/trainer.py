import torch
import torch.nn as nn

def train_model(model, train_loader, criterion, optimizer, num_epochs = 20, valid_loader = None, learning_rate = 0.001):
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

            valid_loss, valid_acc = eval_model(model = model, valid_loader = valid_loader, criterion = criterion, optimizer = optimizer, num_epochs = num_epochs, learning_rate = learning_rate)
            valid_loss_history.append(valid_loss)

            if step_idx % 100 == 0:
                print(f'[Epoch {epoch} / {num_epochs}], step {step_idx} : train loss - {loss}, valid loss - {valid_loss}, valid_acc - {valid_acc * 100:.2f}%')
        
        avg_loss = epoch_loss.item() / len(train_loader)
        train_loss_history.append(avg_loss)

        return train_loss_history, valid_loss_history
    
def eval_model(model, valid_loader, criterion, optimizer, num_epochs = 20, learning_rate = 0.001):
    model.eval()

    correct, total = 0, 0
    loss_list = []

    with torch.no_grad():
        for idx, (source_batch, target_batch) in enumerate(valid_loader):
            pred_batch = model(source_batch, target_batch)
            batch_size, seq_length = target_batch.size()
            loss = criterion(pred_batch.view(batch_size * seq_length, -1), target_batch.view(-1))

            loss_list.append(torch.mean(loss).item())
            print(torch.argmax(pred_batch, dim = 2).shape)
            print(target_batch.shape)
            print(torch.argmax(pred_batch, dim = 2))
            print(target_batch)
            correct += torch.sum((torch.argmax(pred_batch, dim = 2) == target_batch).float())
            print(correct)
            total += target_batch.size(0)


        return sum(loss_list) / len(loss_list), correct / total
            