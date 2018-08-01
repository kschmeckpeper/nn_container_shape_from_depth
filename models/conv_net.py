import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self,
                 input_image_size=128,
                 num_output_channels=128, 
                 num_hidden_channels=1024,
                 stride=2,
                 kernel=5,
                 num_linear_layers=3,
                 dropout_prob=0.0,

                 nonlinearity='ReLU'):
        super(ConvNet, self).__init__()

        layers = OrderedDict()
        
        layers['conv_0'] = nn.Conv2d(1, num_hidden_channels, kernel, stride=stride)

        num_conv_layers = -1
        while input_image_size > 1:
            input_image_size = (input_image_size - kernel - 1) / stride + 1
            num_conv_layers += 1

        for i in range(num_conv_layers):
            layers[nonlinearity + '_conv_' + str(i)] = ConvNet._get_nonlinearity(nonlinearity)
            layers['dropout_conv_' + str(i)] = nn.Dropout(p=dropout_prob)
            layers['conv_' + str(i)] = nn.Conv2d(num_hidden_channels, num_hidden_channels, kernel, stride=stride)


        for i in range(num_linear_layers):
            layers[nonlinearity + str(i)] = ConvNet._get_nonlinearity(nonlinearity)
            
            layers['dropout' + str(i)] = nn.Dropout(p=dropout_prob)
            layers['linear' + str(i)] = nn.Linear(num_hidden_channels, num_hidden_channels)

        layers[nonlinearity + '_final'] = ConvNet._get_nonlinearity(nonlinearity)
        layers['output'] = nn.Linear(num_hidden_channels, num_output_channels)
        self.network = nn.Sequential(layers)


    def forward(self, x):
        return self.network(x) + 1.0

    @staticmethod
    def _get_nonlinearity(nonlinearity):
        ''' Looks up the correct nonlinearity to use
        '''
        if nonlinearity == 'LeakyReLU':
            return nn.LeakyReLU()
        elif nonlinearity == 'ReLU':
            return nn.ReLU()
        else:
            raise NotImplementedError('Invalid nonlinearity')
