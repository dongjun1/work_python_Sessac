import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
import config
from data_handler import generate_dataset, plot_loss_history, modify_dataset_for_ffn

# len(alphabets) * max_length * hidden_size + hidden_size * len(languages)
# 32 * 12 * 64 + 64 * 18 = 25000
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, alphabets, max_length, languages, tunes):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(len(alphabets) * max_length, hidden_size)
        self.layer2 = nn.Linear(hidden_size, len(languages))
    
        if BatchNormalization in tunes:
            self.batch_norm = BatchNormalization(hidden_size)

    def forward(self, x):
        # x: (batch_size, max_length, len(alphabets)) : (32, 12, 57) -> (32, 12*57)
        # FeedForwardNetwork 는 하나의 input만 받을 수 있기 때문.
        output = self.layer1(x)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        output = F.relu(output)
        output = self.layer2(output)
        output = F.log_softmax(output, dim = -1)

        return output # (batch_size, len(languages)) : (32, 18)

    def train_model(self, train_data, valid_data, epochs = 100, learning_rate = 0.001, print_every = 1000):
        criterion = F.nll_loss
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        step = 0
        train_loss_history = []
        valid_loss_history = []

        train_log = {}

        for epoch in range(epochs):
            for x, y in train_data:
                step += 1
                y_pred = self(x)
                loss = criterion(y_pred, y)
                mean_loss = torch.mean(loss).item()


            if step % print_every == 0 or step == 1:

                train_loss_history.append(mean_loss)
                valid_acc, valid_loss = self.evaluate_model(valid_data)
                valid_loss_history.append(valid_loss)
                print(f'[Epoch {epoch}, Step {step}] train_loss : {mean_loss}, valid_loss : {valid_loss}, valid_acc = {valid_acc}')
                torch.save(self, f'checkpoints/FeedForwardNetwork_{step}.chkpts')
                print(f'saved model to checkpoints/FeedForwardNetwork_{step}.chkpts')
                train_log[f'checkpoints/FeedForwardNetwork_{step}.chkpts'] = [valid_loss, valid_acc]

            pickle.dump(train_log, open('checkpoints/train_log.dict', 'wb+'))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        return train_loss_history, valid_loss_history

    def evaluate_model(self, data):
        self.eval()

        criterion = F.nll_loss

        correct, total = 0, 0
        loss_list = []

        with torch.no_grad():
            for x, y in data:
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss_list.append(torch.mean(loss).item())
                correct += torch.sum((torch.argmax(y_pred, dim = 1) == y).float())
                total += y.size(0)

        return correct / total, sum(loss_list) / len(loss_list)
    
if __name__ == '__main__':
    from tuning import BatchNormalization

    tunes = [BatchNormalization]
    train_dataset, valid_dataset, test_dataset, alphabets, max_length, languages  = generate_dataset()
    model = FeedForwardNetwork(32, alphabets, max_length, languages, tunes)
    # acc, loss = model.evaluate_model(train_data)
    
    train_data, valid_data, test_data = modify_dataset_for_ffn(train_dataset, config.batch_size)

    train_loss_history, valid_loss_history = model.train_model(train_data, valid_data)
    
    plot_loss_history(train_loss_history, valid_loss_history, save_dir = f'{model._get_name()}')

    # print(config.graph_dir.join(f'{model._get_name()}'))
    # import os 
    # print(os.path.join(config.graph_dir, model._get_name()))

    # def myjoin(delim, lst):
    #     res = ''
    #     for idx, e in enumerate(lst):
    #         if idx < len(lst) - 1:
    #             res += f'{e}{delim}'
    #         else:
    #             res += e 
    #     return res 
    
    # print(myjoin(' ', ['a', 'b', 'c']))
    