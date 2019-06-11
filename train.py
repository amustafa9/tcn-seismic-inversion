# This script trains the model defined in model file on the seismic offset gathers
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import preprocessing
from tensorboardX import SummaryWriter

from core.utils import *
from core.data_loader import *
from core.model import *
from core.results import *
#%% Fix the random seeds
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


#%% Define function to perform train-val split
def train_val_split(args):
    # Load data
    seismic_offsets = marmousi_seismic().squeeze()[:, 100:600]  # dim= No_of_gathers x trace_length
    impedance = marmousi_model().T[:, 100:600]  # dim = No_of_traces x trace_length

    # Split into train and val
    train_indices = np.linspace(0, 2720, args.n_wells).astype(int)
    val_indices = np.setdiff1d(np.arange(0, 2720).astype(int), train_indices)
    x_train, y_train = seismic_offsets[train_indices], impedance[train_indices]
    x_val, y_val = seismic_offsets[val_indices], impedance[val_indices]

    # Standardize features and targets
    feature_scaler = preprocessing.StandardScaler().fit(x_train)
    target_scaler = preprocessing.StandardScaler().fit(y_train)
    x_train_norm, y_train_norm = feature_scaler.transform(x_train), target_scaler.transform(y_train)
    x_val_norm, y_val_norm = feature_scaler.transform(x_val), target_scaler.transform(y_val)

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm


#%% Define train function
def train(args):
    """
    Sets up the model to train
    """
    # Create a writer object to log events during training
    writer = SummaryWriter(pjoin('runs', 'first_exp'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits
    x_train, y_train, x_val, y_val = train_val_split(args)

    # Convert to torch tensors in the form (N, C, L)
    x_train = torch.from_numpy(np.expand_dims(x_train, 1)).float().to(device)
    y_train = torch.from_numpy(np.expand_dims(y_train, 1)).float().to(device)
    x_val = torch.from_numpy(np.expand_dims(x_val, 1)).float().to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, 1)).float().to(device)

    # Set up the dataloader for training dataset
    dataset = SeismicLoader(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=False)

    # import tcn
    model = TCN(1,
                1,
                args.tcn_layer_channels,
                args.kernel_size,
                args.dropout).to(device)

    # Set up loss
    criterion = torch.nn.MSELoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=0.0001,
                                 lr=0.001)

    # Set up list to store the losses
    train_loss = [np.inf]
    val_loss = [np.inf]
    iter = 0
    # Start training
    for epoch in range(args.n_epoch):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar(tag='Training Loss', scalar_value=loss.item(), global_step=iter)
            if epoch % 20 == 0:
                with torch.no_grad():
                    model.eval()
                    y_pred = model(x_val)
                    loss = criterion(y_pred, y_val)
                    val_loss.append(loss.item())
            print('epoch:{} - Training loss: {:0.4f} | Validation loss: {:0.4f}'.format(epoch,
                                                                                        train_loss[-1],
                                                                                        val_loss[-1]))
        iter += 1

    writer.close()

    # Set up directory to save results
    results_directory = 'results'
    seismic_offsets = np.expand_dims(marmousi_seismic().squeeze()[:, 100:600], 1)
    seismic_offsets = torch.from_numpy((seismic_offsets - seismic_offsets.mean()) / seismic_offsets.std()).float()
    with torch.no_grad():
        model.cpu()
        model.eval()
        AI_inv = model(seismic_offsets)

    if not os.path.exists(results_directory):
        os.mkdir(results_directory)
        print('Saving results...')
    else:
        print('Saving results...')

    np.save(pjoin(results_directory, 'AI.npy'), marmousi_model().T[:, 100:600])
    np.save(pjoin(results_directory, 'AI_inv.npy'), AI_inv.detach().numpy().squeeze())
    print('Results successfully saved.')
        #%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50,
                        help='# of the epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=19,
                        help='Batch size. Default is mini-batch with batch size of 1.')
    parser.add_argument('--tcn_layer_channels', nargs='+', type=int, default=[3, 5, 5, 5, 6, 6],
                        help='No of channels in each temporal block of the tcn.')
    parser.add_argument('--kernel_size', nargs='?', type=int, default=5,
                        help='kernel size for the tcn')
    parser.add_argument('--dropout', nargs='?', type=int, default=0.2,
                        help='Dropout for the tcn')
    parser.add_argument('--n_wells', nargs='?', type=int, default=19,
                        help='# of well-logs used for training')

    args = parser.parse_args()
    train(args)
    evaluate(args)