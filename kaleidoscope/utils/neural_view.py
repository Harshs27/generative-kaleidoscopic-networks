"""
Utils for Generative Kaleidoscopic Networks
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DNN(torch.nn.Module):
    """The DNN architecture to map the input to input.
    """
    def __init__(self, I, H, O, model_type='MLP', image_metadata=[1,28,28], USE_CUDA=False):
        """Initializing the MLP for the regression 
        network.

        Args:
            I (int): The input dimension
            H (int): The hidden layer dimension
            O (int): The output layer dimension
            USE_CUDA (bool): Flag to enable GPU
        """
        super(DNN, self).__init__() # init the nn.module
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        self.I, self.H, self.O = I, H, O
        C, W, H =image_metadata
        if model_type=='MLP': self.MLP = self.getMLP()
        if model_type=='MLP-synthetic': self.MLP = self.getMLP_synthetic()


    def getMLP(self):  # Used for images expts
        l1 = nn.Linear(self.I, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, self.H).type(self.dtype)
        l3 = nn.Linear(self.H, self.H).type(self.dtype)
        l4 = nn.Linear(self.H, self.H).type(self.dtype)
        l5 = nn.Linear(self.H, self.H).type(self.dtype)
        l6 = nn.Linear(self.H, self.H).type(self.dtype)
        l7 = nn.Linear(self.H, self.H).type(self.dtype)
        l8 = nn.Linear(self.H, self.H).type(self.dtype)
        l9 = nn.Linear(self.H, self.H).type(self.dtype)
        l10 = nn.Linear(self.H, self.H).type(self.dtype)
        l11 = nn.Linear(self.H, self.O).type(self.dtype)
        return nn.Sequential(
            l1, nn.ReLU(), 
            l2, nn.ReLU(),  
            l3, nn.ReLU(),
            l4, nn.ReLU(),
            l5, nn.ReLU(), 
            l6, nn.ReLU(),
            l7, nn.ReLU(),
            l8, nn.ReLU(), 
            l9, nn.ReLU(),
            l10, nn.ReLU(), 
            l11,
            # nn.Sigmoid()
            nn.Tanh(),
            ).type(self.dtype)


    def getMLP_synthetic(self):  # Used for 1D & 2D analysis expts
        l1 = nn.Linear(self.I, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, self.H).type(self.dtype)
        l3 = nn.Linear(self.H, self.H).type(self.dtype)
        l4 = nn.Linear(self.H, self.H).type(self.dtype)
        l5 = nn.Linear(self.H, self.H).type(self.dtype)
        l6 = nn.Linear(self.H, self.H).type(self.dtype)
        l7 = nn.Linear(self.H, self.O).type(self.dtype)
        return nn.Sequential(
            l1, nn.ReLU(),
            l2, nn.ReLU(), 
            l3, nn.ReLU(),
            l4, nn.ReLU(),
            l5, nn.ReLU(), 
            l6, nn.ReLU(),
            l7,
            nn.Sigmoid()
            # nn.Tanh()
            ).type(self.dtype)


def get_optimizers(model, lr=0.002, use_optimizer='adam'):
    if use_optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr, 
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    else:
        print('Optimizer not found!')
    return optimizer