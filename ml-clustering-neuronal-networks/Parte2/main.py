import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import utils

from char_rnn import char_rnn  
from predict import sample
from torch import nn
from train import train


def main():
    with open('corpus_espanol.txt', 'r', encoding='ISO-8859-1') as f:
        text = f.read()

    #print(text[:300])

    # encode the text and map each character to an integer and vice versa

    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    # define and print the net
    n_hidden=512
    n_layers=4

    net = char_rnn(chars, n_hidden, n_layers)
    print(net)

    batch_size = 64
    seq_length = 160 #max length verses
    n_epochs = 50 # start smaller if you are just testing initial behavior

    # train the model
    saved_epochs, mean_losses, mean_val_losses = train(net, encoded, epochs=n_epochs,
                                                       batch_size=batch_size, seq_length=seq_length,
                                                       lr=0.001, print_every=1)
    
    for e in saved_epochs:
        # Here we have loaded in a model that trained over e epochs `rnn_e_epoch.net`
        with open('rnn_{}_epoch.net'.format(e), 'rb') as f:
            checkpoint = torch.load(f)
            
        loaded = char_rnn(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        loaded.load_state_dict(checkpoint['state_dict'])
        print("")
        print("Epoch {}:".format(e + 1))

        # Sample epoch
        print(sample(loaded, 1000, prime='<doc ', top_k=5))

    fig, ax = plt.subplots()

    ax.plot([x for x in range(1, n_epochs + 1)], mean_losses, c='r', label='Pérdida en entrenamiento')
    ax.plot([x for x in range(1, n_epochs + 1)], mean_val_losses, c='b', label='Pérdida en validación')
    plt.xticks([x for x in range(1, n_epochs + 1, 2)])
    plt.legend(loc="upper right")

    ax.set(xlabel='épocas', ylabel='función de pérdida',
          title='épocas vs. función de pérdida')
    ax.grid()

    plt.show()

if __name__ == "__main__":
   main()