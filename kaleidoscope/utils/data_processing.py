"""
Data processing and post-processing
functions for Manifold Learning.
"""

from pathlib import Path
from sklearn import preprocessing
from torchvision import datasets, transforms

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import imageio.v3 as imageio
# from pygifsicle import optimize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

import kaleidoscope.main as kals
from kaleidoscope.main import forward_DNN


# ************** Synthetic analysis function *********************
def manifold_sample_analysis_1D(
        model_NN, 
        title='Manifold NN', 
        model_type='MLP-synthetic', 
        image_metadata=[1,1,1], 
        sampling_runs=10,
        norm_range='0to1',
        save_folder=None
    ):
    """
    Runs the following function to analyse the samples
    1. Plots the loss function profile of the model_NN
    2. Apply Kaleidoscope sampling to get K samples
    """
    print(f'Plotting the loss function vs input profile')
    # 100 linearly spaced numbers
    num_pts = 100
    low, high = 0, 1
    if norm_range=='-1to1': low, high = -1, 1 
    x_range = np.linspace(low, high, num_pts).reshape(-1, 1)
    x_range = pd.DataFrame(x_range)
    loss = plot_loss_function_NN(
        model_NN, x_range, title='Loss manifold', model_type=model_type, 
        image_metadata=image_metadata
    )
    print(f'Plotting the output vs input profile')
    fx = plot_output_NN(
        model_NN, x_range, title='Output manifold', model_type=model_type,
         scale_y_axis=True, image_metadata=image_metadata
    )
    print('Apply Kaleidoscope sampling to get K samples')
    recovered_samples, input_noise = kals.kaleidoscopic_sampling_synthetic(
        model_NN,
        NUM_SAMPLES=10, 
        SAMPLING_RUNS=sampling_runs, 
        model_type=model_type, 
        image_metadata=[1,1,1]
    )

    func = x_range, loss
    if save_folder: Path(save_folder).mkdir(parents=True, exist_ok=True)

    for run in recovered_samples.keys():
        points = recovered_samples[run], input_noise
        fname = save_folder + str(run) + '.jpg' if save_folder else None 
        plot_loss_function_and_samples_NN(
            func, points, model_NN, title=f'Kaleidoscopic sampling: run_{run+1}',
            title_font=20, model_type=model_type, image_metadata=image_metadata,
            save_fname=fname 
        )
    return


def manifold_sampling_analysis_2D(
        model_NN,
        title='Manifold MLP',
        model_type='MLP-synthetic',
        image_metadata=[1,1,2],
        SAMPLING_RUNS=30,
        save_folder=None
    ):
    """
    Runs the following function to analyse the samples
    1. Plots the loss function profile of the model_NN
    2. Apply Manifold sampling to get K samples

    """
    print(f'Plotting the loss function profile')
    # 100 linearly spaced numbers
    x_range, y_range = np.meshgrid(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05))
    loss = plot_loss_function_NN_2D(
        model_NN, [x_range, y_range], title='Loss manifold',
        model_type=model_type, image_metadata=image_metadata
    )
    print('Apply Kaleidoscopic sampling to get K samples')
    recovered_samples, input_noise = kals.kaleidoscopic_sampling_synthetic(
        model_NN,
        NUM_SAMPLES=10, 
        SAMPLING_RUNS=SAMPLING_RUNS, 
        model_type=model_type, 
        image_metadata=image_metadata
    )
    if save_folder: Path(save_folder).mkdir(parents=True, exist_ok=True)
    func = [x_range, y_range], loss
    for run in recovered_samples.keys():
        fname = save_folder + str(run) + '.jpg' if save_folder else None 
        points = recovered_samples[run], input_noise
        plot_loss_function_and_samples_NN_2D(
            func, points, model_NN, title=f'Kaleidoscopic sampling: run_{run+1}',
            model_type=model_type, image_metadata=image_metadata,
            save_fname=fname
        )
    return


# ************** plot functions for 1D, 2D ***********************
def plot_loss_vs_epoch(epoch_vs_loss, epoch_vs_test_loss):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'b', label='train')
    plt.plot(epoch_vs_test_loss[:, 0], epoch_vs_test_loss[:, 1], 'o', label='test')
    plt.title('Loss trend', fontsize=30)
    plt.xlabel('epochs', fontsize=25)
    plt.ylabel('loss', fontsize=25)
    plt.legend(loc='upper right', fontsize=20)
    ax.tick_params(axis='both', labelsize=25)
    plt.show()


def plot_xy(x, y, x_label='', y_label='', title='', fig=None, scale_y_axis=False, title_font=30):
    # setting the axes at the centre
    if fig is None: fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('zero') # 'center'
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if scale_y_axis: ax.set_ylim([0, 1])
    # plot the function
    plt.plot(x, y, 'b')
    if len(title)>0: plt.title(title, fontsize=title_font, y=1.05)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    ax.tick_params(axis='both', labelsize=15)
    # fig.tight_layout()


def evaluate_model_loss(model_NN, x, model_type='MLP', image_metadata=[1,1,1], USE_CUDA=False):
    model, _, _ = model_NN
    x = old_convertToTorch(np.array(x), req_grad=False, use_cuda=USE_CUDA)
    fx, _ = forward_DNN(x, model, model_type, image_metadata)
    D = x.shape[-1]
    loss = torch.linalg.norm(x-fx, ord=2, dim=1)**2/D
    # mse = nn.MSELoss()
    # loss = mse(x, fx)
    loss = t2np(loss.cpu())
    return loss


def plot_loss_function_NN(model_NN, x, title='', SAVE=False, scale_y_axis=False, model_type='MLP', image_metadata=[1,1,1]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    loss = evaluate_model_loss(model_NN, x, model_type=model_type, image_metadata=image_metadata)
    plot_xy(x, loss, x_label='input', y_label='loss', title=title, scale_y_axis=scale_y_axis)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return loss


def evaluate_model_output(model_NN, x, model_type='MLP', image_metadata=[1,1,1]):
    model, _, _ = model_NN
    x = convertToTorch(np.array(x), req_grad=False)
    fx, _ = forward_DNN(x, model, model_type=model_type, image_metadata=image_metadata)
    fx = t2np(fx)
    return fx


def plot_output_NN(model_NN, x, title='', SAVE=False, scale_y_axis=False, model_type='MLP', image_metadata=[1,1,1]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    fx = evaluate_model_output(model_NN, x, model_type=model_type, image_metadata=image_metadata)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('zero') # 'center'
    # ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    if scale_y_axis: ax.set_ylim([0, 1])  # UPDATE: Change the scales here. 
    # plot the function
    plt.plot(x, fx, 'b')
    if len(title)>0: plt.title(title, fontsize=30)
    plt.xlabel('input', fontsize=25)
    plt.ylabel('output', fontsize=25)
    ax.tick_params(axis='both', labelsize=25)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return fx


def plot_loss_function_and_samples_NN(func, points, model_NN, title='', save_fname=None, scale_y_axis=False, title_font=30, model_type='MLP', image_metadata=[1,1,1]):
    x, loss = func
    recovered_samples, input_noise = points
    loss_noise = evaluate_model_loss(model_NN, input_noise, model_type=model_type, image_metadata=image_metadata)
    loss_samples = evaluate_model_loss(model_NN, recovered_samples, model_type=model_type, image_metadata=image_metadata)
    plot_xy(x, loss, x_label='Input', y_label='Loss', title=title, scale_y_axis=scale_y_axis, title_font=title_font)
    # Plot the points. Add legend
    plt.scatter(input_noise, loss_noise, color='red', marker='o', s=100, label='noise')
    for pt, l in zip(np.array(input_noise).reshape(-1), loss_noise):
        plt.vlines(x=pt, ymax=l, ymin=0, colors='orange', ls='--', lw=1)
    plt.scatter(recovered_samples, loss_samples, marker='o', s=100, color='green', label='samples')
    plt.legend(loc='upper right', fontsize=20)#, bbox_to_anchor=(1.1, 0.5), labelspacing=3)
    # show the plot
    if save_fname: plt.savefig(save_fname, bbox_inches='tight')#dpi=300)
    plt.show()


def plot_xyz(X, Y, Z, x_label='X', y_label='Y', z_label='Z', title=''):
    # fig = plt.figure(figsize=(5, 5))
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')
    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, alpha=0.7) #rstride=8, cstride=8)
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=-0.1, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-0.2, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
    ax.set(xlim=(-0.2, 1.1), ylim=(-0.0, 1.1), zlim=(-0.1, 1.0))
    # ax.set(xlim=(-0.2, 1.1), ylim=(-0.0, 1.1), zlim=(-2, 2.0))
    plt.title(title, fontsize=30, y=0.97)
    ax.set_xlabel(x_label, fontsize=25, rotation=0, labelpad=15)
    ax.set_ylabel(y_label, fontsize=25, rotation=0, labelpad=15)
    ax.set_zlabel(z_label, fontsize=25, rotation=0, labelpad=-35)
    # plt.zlabel(z_label, fontsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    # ax.view_init(elev=20., azim=-60)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=1.0)
    return ax


def plot_loss_function_NN_2D(model_NN, grid_range, title='', SAVE=False, model_type='MLP', image_metadata=[1,1,2]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    x, y = grid_range
    pt_xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    loss = evaluate_model_loss(model_NN, pt_xy, model_type=model_type, image_metadata=image_metadata)
    loss = loss.reshape(x.shape)
    ax = plot_xyz(x, y, loss, x_label='X', y_label='Y', z_label='loss', title=title)
    ax.view_init(elev=20., azim=-60)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return loss


def plot_loss_function_and_samples_NN_2D(func, points, model_NN, title='', save_fname=None, model_type='MLP', image_metadata=[1,1,2]):
    grid_range, loss = func
    x, y = grid_range
    pt_xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    recovered_samples, input_noise = points
    loss_noise = evaluate_model_loss(model_NN, input_noise, model_type=model_type, image_metadata=image_metadata)
    loss_samples = evaluate_model_loss(model_NN, recovered_samples, model_type=model_type, image_metadata=image_metadata)
    ax = plot_xyz(x, y, loss, x_label='X', y_label='Y', z_label='loss', title=title)
    ax.scatter(
        input_noise['d0'], input_noise['d1'], loss_noise, 
        color='red', marker='o',  label='noise', s=100
    )
    ax.scatter(
        recovered_samples['d0'], recovered_samples['d1'], loss_samples,
        marker='o', s=100, color='green', label='samples'
    )  
    plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 0.9))#, labelspacing=3)
    ax.view_init(elev=20, azim=-60)  # Again as we want to change the viewpoint of scatter
    # show the plot
    if save_fname: plt.savefig(save_fname, bbox_inches='tight')#, dpi=300)
    plt.show()


def create_gif_from_plots(folder='saved_images/', gifname='kals1.gif', sampling_runs=1):
    frames = []
    for _n in range(sampling_runs):
        fname = f'{folder+str(_n)}.jpg'
        if Path(fname).is_file():  # file exists
            frame = imageio.imread(fname)
            frames.append(frame)
    frames = np.stack(frames, axis=0)
    print(frames.shape)
    imageio.imwrite(gifname, frames, fps=3, loop=0)
    # optimize(gifname, "optimized.gif") 
    return


def plot_loss_vs_num_pts_vs_vary_dim(dims, results_dim):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    for dim, res in zip(dims, results_dim):
        _num_pts, _loss = np.log2(res[:, 0]), res[:, 1]
        plt.plot(_num_pts, _loss)
        plt.scatter(_num_pts, _loss, label=f'dim {dim}')
        plt.legend(loc='best', fontsize=25)
    ax.spines['left'].set_position('zero') # 'center'
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', labelsize=25)
    plt.title('Manifold loss profile with varying dimensions', fontsize=25, y=1.03)
    plt.xlabel('[log2 scale] num points (for training)', fontsize=25)
    plt.ylabel('loss range (measured at extreme points)', fontsize=25)
    plt.show()


# ************** Preparing inputs (image analysis ) **************

def get_label_image_dict(dataloader):
    """ Get the input data X for Manifold Learning from the 
    images in the dataloader. For each pixel a naming convention 
    is followed.

    Args:
        dataloader (object): Torch oject iterator
    
    Returns
        image_X (pd.DataFrame): Images(B) x Features(D)
        image_metadata (list): [channel, width, height]
    """
    image_X = {}
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # Convert images to numpy
        images = np.array(images)
        labels = np.array(labels)
        B, C, W, H = images.shape  # batch, channel, row, col
        # Reshape the image
        images = images.reshape(B, -1)
        for img, label in zip(images, labels):
            if label in image_X.keys():
                image_X[label].append(img)
            else:
                image_X[label] = [img]
    # Name the dimensions of the flattened images
    pixel_names = [] 
    for c in range(C):  # Go over the number of channels
        for w in range(W):  # row
            for h in range(H):  # col
                pixel_names.append(
                    ['c'+str(c+1)+'_w'+str(w+1)+'_h'+str(h+1)]
                )
    pixel_names = np.array(pixel_names).reshape(-1)
    # Convert to pandas dataframe format
    for label in image_X.keys():
        image_X[label] = pd.DataFrame(np.vstack(image_X[label]), columns = pixel_names)
    image_metadata = [C, W, H]  # used later for reconstruction
    return image_X, image_metadata


def prepare_image_input_manifold_learning(dataloader):
    """ Get the input data pair (X, G) for NGM from the 
    images in the dataloader. For each pixel a naming 
    convention is followed. 

    Args:
        dataloader (object): Torch oject iterator
    
    Returns
        image_X (pd.DataFrame): Images(B) x Features(D)
        image_metadata (list): [channel, width, height]
    """
    image_X = []
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # Convert images to numpy
        images = np.array(images)
        # print(images.shape, labels.shape)
        B, C, W, H = images.shape  # batch, channel, row, col
        # Reshape the image
        image_X.append(images.reshape(B, -1)) 
    # Name the dimensions of the flattened images
    pixel_names = [] 
    for c in range(C):  # Go over the number of channels
        for w in range(W):  # row
            for h in range(H):  # col
                pixel_names.append(
                    ['c'+str(c+1)+'_w'+str(w+1)+'_h'+str(h+1)]
                )
    pixel_names = np.array(pixel_names).reshape(-1)
    # Convert to pandas dataframe format
    image_X = pd.DataFrame(np.vstack(image_X), columns = pixel_names)
    image_metadata = [C, W, H]  # used later for reconstruction
    return image_X, image_metadata


# *********************************************************************
# ************** PyTorch image data handling **************************
def load_image_data(dataset_name, data_path, batch_size, image_norm='0to1'):
    print(f'Loading {dataset_name} data')
    if dataset_name=='MNIST':
        # The output of torchvision datasets are PILImage images of range [0, 1]. 
        if image_norm=='0to1':
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ])
        elif image_norm=='-1to1':
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        trainset = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )
        testset = datasets.MNIST(
            data_path, train=False, transform=transform
        )
    elif dataset_name=='CIFAR':
        if image_norm=='0to1':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif image_norm=='-1to1':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trainset = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )
        testset = datasets.CIFAR10(
            data_path, train=False, transform=transform
        )
    else:
        print(f'dataset {dataset_name} not available')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    def get_details(dataloader):
        total_batches = 0
        for i, (images, labels) in enumerate(dataloader):
            total_batches += 1
        print(f'Batch: images & labels size {images.shape, labels.shape}')
        print(f'Total batches = {total_batches}')
        return None
    print('Train data details:')
    get_details(trainloader)
    print(f'Test data details:')
    get_details(testloader)
    return trainloader, testloader 


def imshow(img, image_norm='0to1'):  # function to show an image
    if image_norm=='-1to1': # TODO UPDATE
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
    return None


def imshow_save(img, image_norm='0to1', fname=None, title=None):  # function to show an image
    if image_norm=='-1to1': # TODO UPDATE
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    if title: plt.title(title)
    plt.imshow(npimg)
    if fname: plt.savefig(fname, bbox_inches='tight')
    plt.show()
    return None


def visualize_image_dataset(dataloader, display_N=4, image_norm='0to1'):
    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    # show images (Increase grid size?)
    imshow(
        torchvision.utils.make_grid(images[:display_N]), image_norm=image_norm
    )
    # range of values in an image, verify the range 
    print(f'Pixels range (min, max) = {images[0].min(), images[0].max()}')
    return None


def pixel_embeddings_to_images(Xs, image_metadata, display_N=10, seed=None, image_norm='0to1'):
    """ The input image embeddings are converted to images.
    Each embedding is a flattened pixelated representation.

    Args:
        Xs (pd.DataFrame): samples (B) x dimensions (D)
        image_metadata (list): [C, W, H]
    """
    C, W, H = image_metadata
    # for img_embedding in Xs:  # iterate through images
    Xs = np.array(Xs).reshape(Xs.shape[0], C, W, H)
    Xs = convertToTorch(Xs)
    # randomly choose N images to display
    size = min(Xs.shape[0], display_N)
    if seed: np.random.seed(seed)
    idx = np.random.choice(range(Xs.shape[0]), size=size, replace=False)
    imshow(
        torchvision.utils.make_grid(Xs[idx]), image_norm=image_norm
    )
    return None


def tensor_to_images(Xs, image_metadata, display_N=10, seed=None, image_norm='0to1', fname=None, title=None):
    """ The input image embeddings are converted to images.
    Each embedding is a flattened pixelated representation.

    Args:
        Xs (Tensor): samples (B) x dimensions (D)
        image_metadata (list): [C, W, H]
    """
    # randomly choose N images to display
    size = min(Xs.shape[0], display_N)
    if seed: np.random.seed(seed)
    idx = np.random.choice(range(Xs.shape[0]), size=size, replace=False)
    _Xs = Xs[idx].detach().cpu()
    C, W, H = image_metadata
    _Xs = _Xs.reshape(_Xs.shape[0], C, W, H)
    imshow_save(
        torchvision.utils.make_grid(_Xs), image_norm=image_norm, fname=fname, title=title
    )
    return None


def create_gif_from_images(folder='saved_images/', gifname='kals1.gif', sampling_runs=1):
    fname = folder+'init_noise.jpg'
    init_noise = imageio.imread(fname)
    frames = [init_noise]*10
    for _n in range(sampling_runs):
        fname = f'{folder+str(_n)}.jpg'
        if Path(fname).is_file():  # file exists
            frames.append(imageio.imread(fname))
    frames = np.stack(frames, axis=0)
    imageio.imwrite(gifname, frames, fps=15, loop=0)
    # optimize(gifname, "optimized.gif")
    return


# ********************* General data manipulation functions**************
def series2df(series):
    "Convert a pd.Series to pd.Dataframe and set the index as header."
    # Convert the series to dictionary.
    series_dict = {n:v for n, v in zip(series.index, series.values)}
    # Create the dataframe from series and transpose.
    df = pd.DataFrame(series_dict.items()).transpose()
    # Set the index row as header and drop it from values.
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df


def t2np(x):
    "Convert torch to numpy"
    return x.detach().cpu().numpy()


def convertToTorch(data, req_grad=False, device="cpu"):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
    data.requires_grad = req_grad
    return data.to(device)


def old_convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data).type(dtype) # .astype(np.float, copy=False)
    data.requires_grad = req_grad
    return data


def normalize_table(X, method='min_max'):
    """Normalize the input data X.

    Args:
        X (pd.Dataframe): Samples(M) x Features(D).
        methods (str): min_max/mean 

    Returns:
        Xnorm (pd.Dataframe): Samples(M) x Features(D).
        scaler (object): The scaler to scale X
    """
    if method=='min_max':
        scaler = preprocessing.MinMaxScaler()
    elif method=='mean':
        scaler = preprocessing.StandardScaler()
    else: # none
        print(f'Scaler not applied')
        scaler = None
    # Apply the scaler on the data X
    Xnorm = scaler.fit_transform(X) if scaler else X
    # Convert back to pandas dataframe
    Xnorm = pd.DataFrame(Xnorm, columns=X.columns)
    return Xnorm, scaler


def inverse_norm_table(Xnorm, scaler):
    """
    Apply the inverse transform on input normalized
    data to get back the original data.
    """
    return scaler.inverse_transform(Xnorm) if scaler else Xnorm

# ***********************************************************************
