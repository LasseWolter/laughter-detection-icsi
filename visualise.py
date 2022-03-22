from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
import config as cfg
import os
import seaborn as sns
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

def plot_prec_recall_curve(dfs_with_labels, out_name='prec_recall_curve.png', split='', show=False):
    """
    Input: list of tuples of dfs and their labels. 
    The dfs are metrics dataframes containing precisions and recall across all meetings
    
    Plots precision-recall-curves for each dataframe and stores the plot on disk
    """
    
    out_file = os.path.join(cfg.ANALYSIS['plots_dir'], 'prec_recall', f'{split}_{out_name}')

    fig, axs = plt.subplots()

    cols = ['b', 'g', 'r', 'c']
    if len(dfs_with_labels) > len(cols): print('More plots than colours, colours will be repeated')
    for idx, (df, label) in enumerate(dfs_with_labels):
        if 'recall' not in df.columns or 'precision' not in df.columns:
            raise LookupError(
                f'Missing precision or recall column in passed dataframe. Found columns: {df.columns}')

        axs.plot(df['recall'], df['precision'], f'{cols[idx%len(cols)]}--', label=label)
        axs.plot(df['recall'], df['precision'], f'{cols[idx%len(cols)]}o')
    axs.set_ylabel('Precision')
    axs.set_xlabel('Recall')
    axs.set_title(split)
    axs.legend()

    plt.savefig(out_file)
    if(show):
        plt.show()

def plot_conf_matrix(eval_df_path, split, name='conf_matrix', thresholds=[], min_len=None, show=False):
    '''
    Calculate and plot confusion matrix across all meetings per parameter set
    You can specify thresholds(several) and min_len(one) which you want to include
    If nothing passed, all thresholds and min_lens will be plotted
    '''
    path = Path(eval_df_path)
    eval_df = pd.read_csv(path / f"{split}_{cfg.ANALYSIS['eval_df_cache_file']}")
    sum_vals = eval_df.groupby(['threshold', 'min_len'])[['corr_pred_time', 'tot_pred_time', 'tot_transc_laugh_time', 'tot_fp_speech_time', 'tot_fp_noise_time', 'tot_fp_silence_time']].agg(['sum']).reset_index()

    # Flatten Multi-index to Single-index
    sum_vals.columns = sum_vals.columns.map('{0[0]}'.format) 

    # Select certain thresholds and min_len if passed
    if len(thresholds) != 0:
        sum_vals = sum_vals[sum_vals.threshold.isin(thresholds)]
    if min_len != None:
        sum_vals = sum_vals[sum_vals.min_len == min_len]

    print(sum_vals)
    conf_ratio = sum_vals[['corr_pred_time', 'tot_fp_speech_time', 'tot_fp_silence_time', 'tot_fp_noise_time']].copy()
    conf_ratio = conf_ratio.div(sum_vals['tot_pred_time'], axis=0)
    # Set all ratio-vals to 0 if there is no prediction time at all
    conf_ratio.loc[sum_vals.tot_pred_time == 0.0, ['corr_pred_time', 'tot_fp_speech_time','tot_fp_silence_time', 'tot_fp_noise_time']] = 0 


    labels = ['laugh', 'speech', 'silence', 'noise']

    sns.heatmap(conf_ratio, yticklabels=sum_vals['threshold'], xticklabels=labels, annot=True)
    plt.tight_layout()
    plot_file = os.path.join(cfg.ANALYSIS['plots_dir'], 'conf_matrix', f'{name}.png')
    plt.savefig(plot_file)
    
    print('\n=======Confusion Matrix========')
    print(conf_ratio)
    if show:
        plt.show()

# ============================================
# EXPERIMENTS 
# ============================================
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
    plot_train_metrics(few_batches_df, name='more_batches_df', out_dir=out_dir)

    more_batches_df = pd.read_csv('./results/1_to_10_23_02/metrics.csv')
    plot_train_metrics(more_batches_df, name='few_batches_df', out_dir=out_dir)

def compare_prec_recall(dirs_with_labels, min_len, split, show=False):
    '''
    Compare the prec-recall curve of different experiments
    The input should be a list of tuples containing elements of (dir-path, label).
    Each dir-path needs to be a directory ending in 'pred' containing the predictions for each split 
    in a subfolder 'train', 'dev' and 'test'
    - min_len: for which min_len parameter should the prec-recall curve be plotted
    '''
    dfs = []
    for (dir, label) in dirs_with_labels:
        df = pd.read_csv(f"{dir}/{split}_{cfg.ANALYSIS['sum_stats_cache_file']}")
        min_len_df = df[df.min_len == min_len].copy()
        dfs.append((min_len_df, label)) 
    plot_prec_recall_curve(dfs, out_name='compare_class_balance_dev_set', split=split,  show=show)

if __name__ == '__main__':
    # compare_num_of_val_batches()

    # PREC-RECALL
    dirs = [
        './results/1_to_1_21_03/preds',
        './results/1_to_10_16_03/preds',
        './results/1_to_20_16_03/preds',
        './results/1_to_20_struc_22_03/preds/'
     ]
    labels = ['1_to_1', '1_to_10', '1_to_20', 'struc_1_to_20']
    dirs_with_labels = list(zip(dirs, labels))
    compare_prec_recall(dirs_with_labels, min_len=0.2, split='dev', show=True)

    # CONF-MATRIX
    thrs = np.linspace(0,1,11).round(2)
    for dir, label in dirs_with_labels: 
        plot_conf_matrix(dir, split='dev', name=label, thresholds=thrs, min_len=0.2, show=True)

