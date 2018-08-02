import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self,
                 input_image_size=128,
                 num_output_channels=128, 
                 num_hidden_channels=1024,
                 max_pooling_kernel=2,
                 conv_kernel=5,
                 num_linear_layers=3,
                 dropout_prob=0.0,
                 nonlinearity='ReLU'):
        super(ConvNet, self).__init__()
        self.num_hidden_channels = num_hidden_channels
        conv_layers = OrderedDict()
        
        conv_layers['conv_0'] = nn.Conv2d(1, num_hidden_channels, conv_kernel, padding=(conv_kernel-1)/2)
        conv_layers['pooling_0'] = nn.MaxPool2d(max_pooling_kernel)
        num_conv_layers = int(math.ceil(math.log(input_image_size, 2)))

        for i in range(1, num_conv_layers):
            conv_layers[nonlinearity + '_conv_' + str(i)] = ConvNet._get_nonlinearity(nonlinearity)
            conv_layers['dropout_conv_' + str(i)] = nn.Dropout(p=dropout_prob)
            conv_layers['conv_' + str(i)] = nn.Conv2d(num_hidden_channels, num_hidden_channels, conv_kernel, padding=(conv_kernel-1)/2)
            conv_layers['pooling_' + str(i)] = nn.MaxPool2d(max_pooling_kernel)
        self.conv_network = nn.Sequential(conv_layers)
        self.conv_layers = conv_layers

        linear_layers = OrderedDict()
        for i in range(num_linear_layers):
            linear_layers[nonlinearity + '_linear_' + str(i)] = ConvNet._get_nonlinearity(nonlinearity)
            
            linear_layers['dropout_linear_' + str(i)] = nn.Dropout(p=dropout_prob)
            linear_layers['linear_' + str(i)] = nn.Linear(num_hidden_channels, num_hidden_channels)

        linear_layers[nonlinearity + '_final'] = ConvNet._get_nonlinearity(nonlinearity)
        linear_layers['output'] = nn.Linear(num_hidden_channels, num_output_channels)
        self.fully_connected_network = nn.Sequential(linear_layers)



    def forward(self, x):
        #print "input", x.shape
        x = self.conv_network(x)
        #for layer in self.conv_layers:
        #    print "Layer", layer
        #    x = self.conv_layers[layer](x)
        #    print "x", x.shape

        x = x.view(-1, self.num_hidden_channels)
        x = self.fully_connected_network(x)

        return x + 1.0
        

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
