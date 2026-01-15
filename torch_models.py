import logging
import numpy as np
import pdb
import torch
import torch.nn as nn

KERNEL_SIZE = 3
PADDING = 'same' # [valid, same]
PADDING_MODE = 'reflect' # [zeros, reflect] Default: 'zeros'

def get_model(model_type, dimensions, n_hidden_layers=None, residual_layer=None, load_model_path=None):
    if model_type == 'dncnn':
        model = DnCNN(dimensions, 64, n_hidden_layers, residual_layer)

    if load_model_path is not None:
        logging.info(f'Loading {model_type}: {load_model_path}')
        model.load_state_dict(torch.load(load_model_path, map_location='cpu'))

    return model

def get_loss(config_name, loss):
    if loss in ['mae', 'l1']:
        loss = torch.nn.L1Loss()
    elif loss in ['mse', 'l2'] and 'magnitude' in config_name:
        loss = torch.nn.MSELoss()
    return loss

class DnCNN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                    in_features, 
                    out_features, 
                    (KERNEL_SIZE, KERNEL_SIZE),
                    stride=1,
                    padding=PADDING,
                    padding_mode=PADDING_MODE
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                    in_features, 
                    out_features, 
                    (KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE),
                    stride=1,
                    padding=PADDING,
                    padding_mode=PADDING_MODE
            )

    def batch_layer(self, out_features):
        if self.dimensions == 2:
            return nn.BatchNorm2d(out_features)
        elif self.dimensions == 3:
            return nn.BatchNorm3d(out_features)

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer):
        super().__init__()
        self.dimensions = dimensions
        self.residual_layer = residual_layer

        self.in_conv = self.conv_layer(1, n_features)
        self.in_act  = nn.ReLU()
        
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append(self.conv_layer(n_features, n_features))
            self.hidden_layers.append(self.batch_layer(n_features)) #tf: eps: 0.001, momentum: 0.99
            self.hidden_layers.append(nn.ReLU())

        self.out_conv = self.conv_layer(n_features, 1)
    
    def forward(self, tensor):
        x = self.in_conv(tensor)
        x = self.in_act(x)

        for h_layer in self.hidden_layers:
            x = h_layer(x)

        x = self.out_conv(x)

        if self.residual_layer: x = x + tensor

        return x