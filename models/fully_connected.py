import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from helpers import get_nonlinearity

class FullyConnected(nn.Module):
    def __init__(self,
                 num_input_channels=128,
                 num_output_channels=128,
                 num_hidden_channels=524,
                 num_hidden_layers=10,
                 dropout_prob=0.0,
                 use_batch_norm=False,
                 nonlinearity='ReLU'):
        super(FullyConnected, self).__init__()

        if not type(num_hidden_channels) is list:
            num_hidden_channels = [num_hidden_channels] * (num_hidden_layers + 1)


        layers = OrderedDict()
        layers['input'] = nn.Linear(num_input_channels, num_hidden_channels[0])


        for i in range(num_hidden_layers):
            layers[nonlinearity + '_' + str(i)] = get_nonlinearity(nonlinearity)
            layers['dropout' + str(i)] = nn.Dropout(p=dropout_prob)
            layers['linear' + str(i)] = nn.Linear(num_hidden_channels[i], num_hidden_channels[i+1])
            if use_batch_norm:
                layers['norm' + str(i)] = nn.BatchNorm(num_hidden_channels)

        layers['final_' + nonlinearity] = get_nonlinearity(nonlinearity)
        layers['output'] = nn.Linear(num_hidden_channels[num_hidden_layers], num_output_channels)
        self.network = nn.Sequential(layers)

    def forward(self, x):
        return self.network(x) + 1.0