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
    split = 'val'

    
    smooth_x, smooth_y = smooth(df.batch_num, df.val_prec, 50)
    # plt.plot(df['batch_num'], df[f'{split}_prec'], 'b--')
    # plt.plot(df['batch_num'], df[f'{split}_rec'],  'r--')
    # plt.plot(df['batch_num'], df[f'{split}_acc'], 'c--')
    # plt.plot(df['batch_num'], df[f'{split}_loss'], 'g--')
    plt.plot(smooth_x,smooth_y, 'c--')
    plt.ylabel('Performance')
    plt.xlabel('Processed batches')
    plt.show()


if __name__ == '__main__':
    metrics_df = pd.read_csv('./results/1_to_1_15_02/metrics.csv')
    plot_train_metrics(metrics_df)
