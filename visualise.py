from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np

def smooth(x, y, num_of_datapoints):
    '''
    Takes x and y values and a number of datapoints the data should be reduced to 
    Returns the new x and y values for the smoothed data
    '''
    x_new=  np.linspace(x.min(), x.max(), num_of_datapoints)
    spl = make_interp_spline(x, y, k=3)
    smooth_y = spl(x_new)
    return(x_new, smooth_y)


def plot_train_metrics(df):
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

    fig, axs = plt.subplots(2,2)

    
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

    plt.show()


if __name__ == '__main__':
    metrics_df = pd.read_csv('./train_results/tmp/metrics.csv')
    plot_train_metrics(metrics_df)
