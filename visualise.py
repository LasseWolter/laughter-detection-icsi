from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
import config as cfg
import os
from pathlib import Path

def smooth(x, y, num_of_datapoints):
    '''
    Takes x and y values and a number of datapoints the data should be reduced to 
    Returns the new x and y values for the smoothed data
    '''
    x_new=  np.linspace(x.min(), x.max(), num_of_datapoints)
    spl = make_interp_spline(x, y, k=3)
    smooth_y = spl(x_new)
    return(x_new, smooth_y)


def plot_train_metrics(df, name='metrics_plot', out_dir=cfg.ANALYSIS['plots_dir'], show=False):
    """
    Input: dataframe containing metrics recorded during training
    Plots these metrics against the number of batches to see how precision, recall, accuracy and loss devloped 

    Columns of df are expected to be:
    [ batch_num,train_prec,train_rec,train_acc,train_loss,val_prec,val_rec,val_acc,val_loss ] 
    """

    # Tupel of split name and the plt representation that should be used for this split
    splits = [('train', 'b--'),('val', 'r--')]
    num_train_samples = 91000 
    batch_size = 32
    batches_per_epoch = num_train_samples/float(batch_size)
    np.linspace(0,num_train_samples*4, 4)

    # Functions used to convert between primary and secondary axis
    def _to_epoch(x):
        return x / batches_per_epoch 

    def _to_batch_num(x):
        return x* batches_per_epoch 

    fig, axs = plt.subplots(2,2, figsize=(15, 10))

    
    for split,plt_rep in splits:
        axs[0,0].plot(df['batch_num'], df[f'{split}_prec'], plt_rep, label=f'{split}_precision')
        axs[0,1].plot(df['batch_num'], df[f'{split}_rec'],  plt_rep, label=f'{split}_recall')
        axs[1,0].plot(df['batch_num'], df[f'{split}_acc'], plt_rep, label=f'{split}_accuracy')
        axs[1,1].plot(df['batch_num'], df[f'{split}_loss'], plt_rep, label=f'{split}_loss')

    axs[0,0].set_title('Precision')
    axs[0,1].set_title('Recall')
    axs[1,0].set_title('Accuracy')
    axs[1,1].set_title('Loss')
    for ax in axs.reshape(-1):
        ax.legend()
        ax.set_ylabel('Performance')
        ax.set_xlabel('Processed batches')
        secax = ax.secondary_xaxis('top', functions=(_to_epoch, _to_batch_num))
        secax.set_xlabel('epochs')

    fig.suptitle('Metrics on train and dev set during training')
    fig.tight_layout(pad=0.3)


    plots_dir = Path(out_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f'{name}.png')

    if(show): plt.show()


def compare_num_of_val_batches():
    '''
    Training the same model with varied number of online validation batches evaluated 
    few_batches_df: 10 batches per validation -> one validation per 200 training batches 
    more_batches_df: 300 batches per validation -> one validation per 900 training batches
    '''
    # Directory for storing plots - create if doesn't exist
    out_dir = os.path.join(cfg.ANALYSIS['plots_dir'],'compare_batch_num')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Create visualisation for few and more batches 
    few_batches_df= pd.read_csv('./results/1_to_10_new_val_23_02/metrics.csv')
    plot_train_metrics(few_batches_df, name='few_batches_df', out_dir=out_dir)

    more_batches_df = pd.read_csv('./results/1_to_10_23_02/metrics.csv')
    plot_train_metrics(more_batches_df, name='more_batches_df', out_dir=out_dir)

if __name__ == '__main__':
    compare_num_of_val_batches()
    df = pd.read_csv('./results/1_to_10_new_val_23_02/metrics.csv')
    plot_train_metrics(df)
