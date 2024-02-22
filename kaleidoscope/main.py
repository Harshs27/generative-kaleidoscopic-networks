"""
Functions for Generative Kaleidoscopic Networks
1. Manifold Learning
2. Kaleidoscopic Sampling 

Owner: Harsh Shrivastava
Contact: harshshrivastava111@gmail.com
"""

from pathlib import Path

import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
import itertools
import sys
import torch
import torch.nn as nn
import torchvision

# local imports
import kaleidoscope.utils.neural_view as neural_view
import kaleidoscope.utils.data_processing as dp


######################################################################
# Images: Manifold learning & Kaleidoscopic sampling
######################################################################

def forward_DNN(X, model, model_type='MLP', image_metadata=None):
    # 1. Running the NN model, X = B x D
    if model_type in ['MLP', 'MLP-synthetic', 'Transformer']:
        Xp = model.MLP(X)
    elif model_type in ['CNN', 'CNN-bottleneck', 'UNet']:
        Xp = model.CNN(X.reshape(-1, *image_metadata))
        Xp = Xp.reshape(Xp.shape[0], -1)
    else:
        print(f'model type {model_type} is not defined')
    # 2. Calculate the regression loss
    mse = nn.MSELoss() 
    reg_loss = mse(Xp, X)
    return Xp, reg_loss


def set_model_grad(model, req_grad=True):
    for _name, _param in model.named_parameters():
        _param.requires_grad = req_grad


def manifold_learning_image(
    X,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    NORM=None,
    VERBOSE=True, 
    BATCH_SIZE=None,
    USE_CUDA=None,   
    image_metadata=[1, 28, 28], 
    model_type='MLP', 
    pretrained_model=None,
    image_norm='0to1'  # '-1to1'
    ):
    """Fit a NN to learn the data representation from X->X. 
    Return the learned model representing the Manifold. 

    Args:
        X (pd.DataFrame): Samples(M) x Features(D).
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        NORM (str): min_max/mean/None
        VERBOSE (bool): if True, prints to output.
        BATCH_SIZE (int): If None, take all data
        USE_CUDA (str): None/"cuda:x"
        
    Returns:
        model_NN (list): [
            model (torch.nn.object): A NN model,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """ 
    # Select the device for training
    # While using USE_CUDA_DEVICES from cmd, change USE_CUDA="cuda"
    device = torch.device(USE_CUDA) if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    feature_means = X.mean()
    print(f'Means of selected features {feature_means, len(feature_means)}')
    # Normalize the data
    print(f'Normalizing the data: {NORM}')
    X, scaler = dp.normalize_table(X, NORM)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False, device=device)
    M, D = X.shape
    # Initialize the MLP model
    if pretrained_model: 
        if VERBOSE: print(f'Using the Pre-trained model as initial seed')
        model, scaler, feature_means = pretrained_model
        model.train()
        for _name, _param in model.named_parameters():
            _param.requires_grad = True
            print(_name, _param.requires_grad)
    else:
        if VERBOSE: print(f'Initializing a DNN model')
        model = neural_view.DNN(
            I=D, H=hidden_dim, O=D, 
            model_type=model_type, image_metadata=image_metadata
        )
    model = model.to(device)
    optimizer = neural_view.get_optimizers(model, lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Manifold model initialized {model} with params {total_params}')
    # Defining optimization & model tracking parameters
    min_loss = np.inf
    PRINT = int(epochs/10) # will print only 10 times
    epoch_vs_loss = []  # [(epoch, loss), ...]
    X_batch = X
    for e in range(epochs):
        if not e%PRINT:
            print(f'epoch {e}')
        # Stochastic sampling of learning datapoints
        if BATCH_SIZE:
            X_batch = X[np.random.choice(len(X), BATCH_SIZE, replace=False)]
        # reset the grads to zero
        optimizer.zero_grad()
        # calculate the loss for train data
        Xp, loss = forward_DNN(X_batch, model, model_type, image_metadata) 
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer.step()
        loss = dp.t2np(loss)
        if not e%PRINT and VERBOSE: 
            with torch.no_grad(): # prediction on test 
                model.eval()
                print(f'\n epoch:{e}/{epochs}, loss={loss}')
                print(f'Train images {X_batch.shape}') 
                disp_seed = np.random.choice(1000, size=1)
                dp.tensor_to_images(
                    X_batch, image_metadata, display_N=5, seed=disp_seed, image_norm=image_norm
                )
                # Visualize some test images
                print(f'Recovered images {Xp.shape}')
                dp.tensor_to_images(
                    Xp, image_metadata, display_N=5, seed=disp_seed, image_norm=image_norm
                )
            model.train()
        if e==0 or e==epochs-1 or e%100==99 or (not e%PRINT):
            # EVERY 100th epoch, save the best model.
            if loss < min_loss: 
                print(f'Updating the best model')
                best_model = copy.deepcopy(model)
                set_model_grad(best_model, req_grad=False)
                min_loss = loss
        # Record the loss trend for analysis
        epoch_vs_loss.append([e, loss])
    if VERBOSE: 
        with torch.no_grad(): # prediction on test 
            model.eval()
            print('\n Train data fit')
            dp.tensor_to_images(Xp, image_metadata, display_N=48, image_norm=image_norm)
    epoch_vs_loss = np.array(epoch_vs_loss)
    dp.plot_xy(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'epochs', 'loss')
    return [best_model, scaler, feature_means]


def get_kaleidoscopic_samples(
        model, 
        NUM_SAMPLES=1000, 
        SAMPLING_RUNS=300, 
        model_type='MLP', 
        image_metadata=[1,28,28],
        SHOW_EVERY=1,
        NOISE='uniform',
        USE_CUDA=False, 
        image_norm='0to1',
        eps=0.01, 
        folder=None  # 'saved_images/'
    ):
    if folder: Path(folder).mkdir(parents=True, exist_ok=True)
    # Select the device for training
    # While using USE_CUDA_DEVICES, change USE_CUDA="cuda"
    device = torch.device(USE_CUDA) if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    C, W, H = image_metadata
    if image_norm=='0to1':
        mean, std = 0.5, 0.5 
        low, high = 0, 1
    elif image_norm=='-1to1':
        mean, std = 0, 1
        low, high = -1, 1
    if NOISE=='normal':
        X_in = np.clip(np.random.normal(mean, std, size=[NUM_SAMPLES, C*W*H]), low, high)
    elif NOISE=='uniform':
        X_in = np.clip(np.random.uniform(low, high, size=[NUM_SAMPLES, C*W*H]), low, high)
    X_in = dp.convertToTorch(np.array(X_in), req_grad=False, device=device)
    with torch.no_grad(): 
        model = model.to(device)
        model.eval()
        for _name, _param in model.named_parameters():
            _param.requires_grad = False
        disp_seed = np.random.choice(1000, size=1)[0]
        # Saving the input noise
        title = f'Initial Noise'
        fname = folder + 'init_noise.jpg' if folder else None
        dp.tensor_to_images(
            X_in, image_metadata, display_N=48, seed=disp_seed, 
            image_norm=image_norm, fname=fname, title=title
        )
        for r in range(SAMPLING_RUNS):
            X_in, _ = forward_DNN(
                X_in, model, model_type, image_metadata
            )
            if r%SHOW_EVERY==0:
                title = f'Sampling run {r}/{SAMPLING_RUNS}'
                fname = folder + str(r) + '.jpg' if folder else None
                dp.tensor_to_images(
                    X_in, image_metadata, display_N=48, seed=disp_seed, 
                    image_norm=image_norm, fname=fname, title=title
                )
            # UPDATE: Adding slight normal noise to each sample to facilitate local minima jump
            if NOISE=='normal':
                noise = torch.normal(mean, std, size=X_in.shape) # normal
            elif NOISE=='uniform':
                noise = (high - low) * torch.rand(X_in.shape) + high  # Uniform
            X_in = X_in + eps*noise.to(device) 
            X_in = torch.clamp(X_in, min=low, max=high)
    return X_in



######################################################################
# Synthetic: Manifold learning & Kaleidoscopic sampling
######################################################################

def manifold_learning_MLP(
    X,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    NORM=None,
    model_type='MLP',
    VERBOSE=True, 
    BATCH_SIZE=None,
    USE_CUDA=True, 
    pretrained_model=None, 
    ):
    """Manifold learning using MLP. 
    1. Fit a MLP to learn the data representation from X->X. 
    2. Return the learned model representing the model

    Args:
        X (pd.DataFrame): Samples(M) x Features(D).
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        VERBOSE (bool): if True, prints to output.
        BATCH_SIZE (int): If None, take all data.
        USE_CUDA (bool): If True, use GPU. 
        pretrained_model (object): if input model, use it as init. 
        
    Returns:
        List: [
            model (torch.nn.object): A MLP model after manifold learning,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """
    # Select the device for training
    device = torch.device("cuda") if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    feature_names = X.columns
    feature_means = X.mean()
    X, scaler = dp.normalize_table(X, NORM)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False).to(device)
    M, D = X.shape
    if VERBOSE: print(f'Data is in {device}, grad is False: {X.requires_grad}')
    # Initialize the MLP model
    if pretrained_model: 
        if VERBOSE: print(f'Using the Pre-trained model as initial seed')
        model, scaler, feature_means = pretrained_model
        model = model.to(device)
        model.train()
        for _name, _param in model.named_parameters():
            _param.requires_grad = True
            print(_name, _param.requires_grad)
    else:
        if VERBOSE: print(f'Initializing a MLP model')
        model = neural_view.DNN(I=D, H=hidden_dim, O=D, model_type=model_type)
        model = model.to(device)
    # Defining optimization & model tracking parameters
    optimizer = neural_view.get_optimizers(model, lr=lr)
    if VERBOSE: print(f'MLP model initialized {model}')
    PRINT = int(epochs/10)+1 # will print only 10 times
    min_loss = np.inf
    epoch_vs_loss = []  # [(epoch, loss), ...]
    for e in range(epochs):
        # Stochastic sampling of learning datapoints
        X_batch = X
        if BATCH_SIZE:
            X_batch = X[np.random.choice(len(X), BATCH_SIZE, replace=False)]         
        # reset the grads to zero
        optimizer.zero_grad()
        # calculate the optimization loss
        Xp, loss = forward_DNN(X_batch, model, model_type=model_type) 
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer.step()
        loss = loss.detach()
        # Printing output
        if not e%PRINT and VERBOSE: 
            # curr_Xp = pd.DataFrame(dp.t2np(Xp), columns=feature_names)
            print(f'\n epoch:{e}/{epochs}, loss={dp.t2np(loss)}')#, val={dp.t2np(Xp)}')
        if e==0 or e==epochs-1 or e%100==99 or (not e%PRINT):
            # EVERY 100th epoch, save the best model.
            if loss < min_loss: 
                best_model = copy.deepcopy(model)
                min_loss = loss
        # Record the loss trend for analysis
        epoch_vs_loss.append([e, dp.t2np(loss)])
    if VERBOSE: print('\n')
    epoch_vs_loss = np.array(epoch_vs_loss)
    dp.plot_xy(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'epochs', 'loss')
    return [best_model, scaler, feature_means]


def kaleidoscopic_sampling_synthetic(
    model_NN,
    NUM_SAMPLES=1000, 
    SAMPLING_RUNS=5, 
    model_type='MLP', 
    image_metadata=[1,28,28],
    USE_CUDA=False, 
    norm_range='0to1'  # '-1to1'
    ):
    device = torch.device(USE_CUDA) if USE_CUDA else torch.device("cpu") 
    # Get the NN params
    model, scaler, feature_means = model_NN
    feature_names = feature_means.index
    C, W, H = image_metadata
    if norm_range=='0to1':
        mean, std = 0.5, 0.5
        low, high = 0, 1
    if norm_range=='-1to1':
        mean, std = 0, 1 
        low, high = -1, 1
    X_in = np.clip(np.random.normal(mean, std, size=[NUM_SAMPLES, C*W*H]), low, high)
    input_noise = pd.DataFrame(X_in, columns=feature_names)
    X_in = dp.convertToTorch(np.array(X_in), req_grad=False, device=device)
    with torch.no_grad(): 
        model.eval()
        recovered_samples = {}
        for r in range(SAMPLING_RUNS):
            print(f'kaleioscopic sampling run: {r}/{SAMPLING_RUNS}')
            X_in, _ = forward_DNN( 
                X_in, model, model_type, image_metadata
            )
            recovered_samples[r] = pd.DataFrame(np.array(X_in), columns=feature_names)
    return recovered_samples, input_noise


def loss_vs_num_pts_expt(dim=1):
    # Make sure the at the loss of training goes extremely low -> 0
    loss_vs_num_pts_plot = []
    for num_pts in [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048]:
        # initialize the points uniformly at random
        X_train = np.random.uniform(0, 1, size=(num_pts, dim))
        dim_names = ['d'+str(i) for i in range(dim)]
        X_train = pd.DataFrame(X_train, columns=dim_names)
        # Initialize a model for manifold learning
        print(f'Training the model with input dim={dim}')
        model_MLP = manifold_learning_MLP(
            X_train,
            hidden_dim=50, 
            epochs=10000,
            lr=0.001,
            BATCH_SIZE=min(1000, X_train.shape[0]),
            model_type='MLP-synthetic',
            NORM=None,
            USE_CUDA=True, 
            pretrained_model=None, 
        )
        # Get the loss values over the extremities of the space grid
        # Given the dimension, say d=2, extremeties will be (0,0), (0,1), (1,0), (1,1)
        extreme_points = np.array([list(i) for i in itertools.product([0, 1], repeat=dim)])
        extreme_points = pd.DataFrame(extreme_points, columns=dim_names)
        print(f'Evaluation points {extreme_points, extreme_points.shape}')
        # evaluate the loss at the extreme points
        loss = dp.evaluate_model_loss(model_MLP, extreme_points, USE_CUDA=True)
        # Get the max loss value. 
        max_loss = max(loss)
        loss_vs_num_pts_plot.append([num_pts, max_loss])
    # plot the curve
    loss_vs_num_pts_plot = np.array(loss_vs_num_pts_plot)
    _num_pts = loss_vs_num_pts_plot[:, 0]
    _loss = loss_vs_num_pts_plot[:, 1]
    # dp.plot_xy(_num_pts, _loss, x_label='num points', y_label='loss range', title=f'Dimension {dim}')
    return loss_vs_num_pts_plot

# ********************************************************************
