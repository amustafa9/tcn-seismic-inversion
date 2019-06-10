# This script generates results from the data stored in the results directory
import os
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#%%



def evaluate():
    """
    Function makes plots on the results and also calculates pcc and r2 scores on both the validation and training data
    """
    
    # Load data
    AI = np.load(pjoin('results', 'AI.npy'))
    AI_inv = np.load(pjoin('results', 'AI_inv.npy')) * AI.std() + AI.mean()

    # Make AI, predicted AI plots
    make_plots()

    # Make scatter plot
    scatter_plot()

    # Make trace plot
    trace_plot()



def plot(img, cmap='rainbow', cbar_label=r'AI ($m/s\times g/cm^3$)', vmin=None, vmax=None):
    Y, X = np.mgrid[slice(0.47, 2.8 + dt, dt), slice(0, 17000 + dx, dx)]

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    if (vmin is None or vmax is None):
        plt.pcolormesh(X, Y, img.T, cmap=cmap)
    else:
        plt.pcolormesh(X, Y, img.T, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar()
    plt.ylabel("Depth (Km)", fontsize=30)
    plt.xlabel("Distance (m)", fontsize=30, labelpad=15)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position("top")
    plt.gca().set_xticks(np.arange(0, 17000 + 1, 1700 * 2))
    plt.tick_params(axis='both', which='major', labelsize=30)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(cbar_label, rotation=270, fontsize=30, labelpad=40)
    return fig


#%% Section figures
def make_plots():
    dt = 0.00466
    dx = 6.25

    vmin = min([AI.min(), AI_inv.min()])
    vmax = max([AI.max(), AI_inv.max()])


    fig = plot(AI, vmin=vmin, vmax=vmax)
    fig.savefig('AI.png', bbox_inches='tight')
    fig = plot(AI_inv, vmin=vmin, vmax=vmax)
    fig.savefig('AI_inv.png', bbox_inches='tight')
    fig = plot(abs(AI_inv - AI), vmin=vmin, vmax=vmax)
    fig.savefig('difference.png', bbox_inches='tight')


def scatter_plot():
    AI = np.expand_dims(AI, axis=1)
    AI_inv = np.expand_dims(AI_inv, axis=1)

    sns.set(style="whitegrid")
    fig = plt.figure()
    np.random.seed(30)
    inds = np.random.choice(AI.shape[0], 30)
    x = np.reshape(AI[inds, 0], -1)
    y = np.reshape(AI_inv[inds, 0], -1)

    std = AI[:, 0].std()

    max = np.max([AI[:, 0].max(), AI_inv[:, 0].max()])
    min = np.min([AI[:, 0].min(), AI_inv[:, 0].min()])

    d = {'True AI': x, 'Estimated AI': y}
    df = pd.DataFrame(data=d)

    fig = plt.figure(figsize=(15, 15))
    g = sns.jointplot("Estimated AI", "True AI", data=df, kind="reg",
                      xlim=(min, max), ylim=(min, max), color="k", scatter_kws={'s': 10}, label='big', stat_func=None)

    plt.xlabel(r"Estimated AI", fontsize=30)
    plt.ylabel(r"True AI", fontsize=30)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('Scatter.png', bbox_inches='tight')
    plt.show()





#
# #%% Trace plots
def trace_plot():
    AI = np.expand_dims(AI, axis=1)
    AI_inv = np.expand_dims(AI_inv, axis=1)
    x_loc = np.array(
        [3400, 6800, 10200, 13600])  ## Choose 4 number betweeb 0 and 17000 (distance in meters along the horizontal)
    inds = (AI.shape[0] * (x_loc / 17000)).astype(int)

    x = AI[inds].squeeze()
    y = AI_inv[inds].squeeze()
    time = np.linspace(0.47, 2.8, 500)
    fig, ax = plt.subplots(1, x.shape[0], figsize=(10, 12), sharey=True)

    max = np.max([y.max(), x.max()]) * 1.2
    min = np.min([y.min(), x.min()]) * 0.8

    for i in range(len(inds)):
        p1 = ax[i].plot(x[i], time, 'k')
        p2 = ax[i].plot(y[i], time, 'r')
        ax[i].set_xlabel(r'AI($m/s \times g/cm^3$)' + '\n' + r'$distance={}m$'.format(x_loc[i]), fontsize=15)
        if i == 0:
            ax[i].set_ylabel('Depth (Km)', fontsize=20)
            ax[i].yaxis.set_tick_params(labelsize=20)

        ax[i].set_ylim(time[0], time[-1])
        ax[i].set_xlim(min, max)
        ax[i].invert_yaxis()
        ax[i].xaxis.set_tick_params(labelsize=10)

    fig.legend([p1[0], p2[0]], ["True AI", "Estimated AI for SVR"], loc="upper center", fontsize=20,
               bbox_to_anchor=(0.5, 1.07))
    plt.show()
    fig.savefig('AI_traces_svr.png'.format(x_loc), bbox_inches='tight')







# Compute Pearson Correlation coefficients and r2 coeffs for training and test traces
train_indices = np.arange(0, 2721, 150).tolist()  # 150 is the sampling interval I used
val_indices = np.arange(0, 2721).tolist()
val_indices = np.setdiff1d(val_indices, train_indices)

pcc_train = 0
r2_train = 0
for i in range(len(train_indices)):
    trace_pred = predicted[train_indices[i]]
    trace_actual = actual[train_indices[i]]
    pcc_train += np.corrcoef(trace_actual, trace_pred)[0, 1]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(trace_actual, trace_pred)
    r2_train += r_value**2
pcc_train = pcc_train/len(train_indices)
r2_train = r2_train/len(train_indices)

pcc_val = 0
r2_val = 0
for i in range(len(val_indices)):
    trace_pred = predicted[val_indices[i]]
    trace_actual = actual[val_indices[i]]
    pcc_val += np.corrcoef(trace_actual, trace_pred)[0, 1]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(trace_actual, trace_pred)
    r2_val += r_value ** 2
pcc_val = pcc_val/len(val_indices)
r2_val = r2_val/len(val_indices)


# Now to compute the losses
tmp1 = (predicted[train_indices] - actual[train_indices])**2
train_loss = np.sum(tmp1) / len(tmp1.flatten())

tmp2 = (predicted[val_indices] - actual[val_indices])**2
val_loss = np.sum(tmp2) / len(tmp2.flatten())

