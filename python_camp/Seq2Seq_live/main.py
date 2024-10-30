import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_handler import parse_file, simple_samples
from rnn_cells import RNNCellManual, LSTMCellManual
from trainer import train_model
from seq2seq import *

if __name__ == '__main__':
    
    path = 'kor-eng/kor.txt'
    eng_fra = 'data/eng-fra.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
<<<<<<< HEAD
    embbeding_dim = 256
    batch_size = 128
=======
    embbeding_dim = 256
    batch_size = 128
>>>>>>> af78c19a8c7b7f88048402c3db5759f5c18f501f
    encoder_model = RNNCellManual
    decoder_model = RNNCellManual
    hidden_dim = 128
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    learning_rate = 0.002
    num_epochs = 50

    

    print(f'Using device is {device}')
    # (train, valid, test), source_vocab, target_vocab = parse_file(path, device, batch_size = batch_size)
    # (train_fra, valid_fra, test_fra), fra_source_vocab, fra_target_vocab = parse_file(eng_fra, device, batch_size = batch_size)
    # print(f'source : {fra_source_vocab.vocab_size}')
    # print(f'target : {fra_target_vocab.vocab_size}')
    (simple_train, simple_valid, simple_test), simple_source_vocab, simple_target_vocab = simple_samples(eng_fra, device, batch_size = batch_size)
    print(f'source vocab : {simple_source_vocab.vocab_size}')
    print(f'target vocab : {simple_target_vocab.vocab_size}')

    # for k, v in target_vocab.idx2word.items():
    #     print(k, v)
    attention = 'LuongAttention'
    # encoder = Encoder(device, fra_source_vocab, embbeding_dim, hidden_dim, encoder_model).to(device)
    # decoder = Decoder(device, fra_target_vocab, embbeding_dim, hidden_dim, decoder_model, attention = attention).to(device)
    encoder = Encoder(device, simple_source_vocab, embbeding_dim, hidden_dim, encoder_model).to(device)
    decoder = Decoder(device, simple_target_vocab, embbeding_dim, hidden_dim, decoder_model, attention = attention).to(device)

    model = Seq2Seq(device = device, encoder = encoder, decoder = decoder).to(device)

    # train_loss_history, valid_loss_history = train_model(model = model, device = device, train_loader = train, valid_loader = valid, 
    #                                                      source_vocab = source_vocab, target_vocab = target_vocab,
    #                                                      criterion = criterion, optimizer = optimizer, num_epochs = num_epochs, learning_rate = learning_rate, 
    #                                                      print_ = True)

    # train_loss_history, valid_loss_history = train_model(model = model, device = device, train_loader = train_fra, valid_loader = valid_fra, 
    #                                                      source_vocab = fra_source_vocab, target_vocab = fra_target_vocab,
    #                                                      criterion = criterion, optimizer = optimizer, num_epochs = num_epochs, learning_rate = learning_rate, 
    #                                                      print_ = True)
    
    train_losses, valid_losses, valid_metrics = train_model(model = model, device = device, train_loader = simple_train, 
                                                         source_vocab = simple_source_vocab, target_vocab = simple_target_vocab,
                                                         criterion = criterion, optimizer = optimizer, num_epochs = num_epochs, learning_rate = learning_rate, 
                                                         print_ = True)
    
    fig, axs = plt.subplots(2,1)

    axs[0].plot(train_losses)
    axs[1].plot(valid_losses)
    
    plt.show()