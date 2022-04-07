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


def plot_train_metrics(df_dir, name='metrics_plot', sub_dir='', show=False):
    """
    Input: path to a .csv-file representing metrics recorded during training
    Plots these metrics against the number of batches to see how precision, recall, accuracy and loss devloped 

    Columns of df are expected to be:
    [ batch_num,train_prec,train_rec,train_acc,train_loss,val_prec,val_rec,val_acc,val_loss ] 
    """
    if not show:
        plt.clf()

    metrics_file = os.path.join(df_dir,'metrics.csv')
    if not os.path.isfile(metrics_file):
        print(f'\nMetrics file not found: {metrics_file}\nSkipping this plot...')
        return

    print(f"\nCreating train-metrics plot for: {df_dir}")
    df = pd.read_csv(os.path.join(df_dir,'metrics.csv'))
    # Tupel of split name and the plt representation that should be used for this split
    splits = [('train', 'b--'),('val', 'r--')]

    if os.path.isfile(os.path.join(df_dir,'train_params.csv')):
        train_params = pd.read_csv(os.path.join(df_dir,'train_params.csv'))
        num_train_samples = train_params.train_samples[0]
        batch_size = 32
        batches_per_epoch = num_train_samples/float(batch_size)
        know_batches_per_epoch=True
    else: 
        print("Couldn't load num_train_samples per epoch. Not displaying secondary axis with epochs.")
        know_batches_per_epoch = False
        batches_per_epoch = 0

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
        if(know_batches_per_epoch):
            secax = ax.secondary_xaxis('top', functions=(_to_epoch, _to_batch_num))
            secax.set_xlabel('epochs')

    fig.suptitle('Metrics on train and dev set during training')
    fig.tight_layout(pad=0.3)

    out_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'train_metrics', f'{name}.png')
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_file)

    if(show): plt.show()

def plot_prec_recall_curve(dfs_with_labels, out_name='prec_recall_curve.png', sub_dir='', split='', show=False):
    """
    Input: list of tuples of dfs and their labels. 
    The dfs are metrics dataframes containing precisions and recall across all meetings
    
    Plots precision-recall-curves for each dataframe and stores the plot on disk
    """
    if not show:
        plt.clf() 

    fig, axs = plt.subplots()

    cols = ['b', 'g', 'r', 'c', 'k', 'y', 'p']
    if len(dfs_with_labels) > len(cols): print('More plots than colours, colours will be repeated')
    for idx, (df, label) in enumerate(dfs_with_labels):
        if 'recall' not in df.columns or 'precision' not in df.columns:
            raise LookupError(
                f'Missing precision or recall column in passed dataframe. Found columns: {df.columns}')

        axs.plot(df['recall'], df['precision'], f'{cols[idx%len(cols)]}--', label=label, alpha=0.5)
        axs.plot(df['recall'], df['precision'], f'{cols[idx%len(cols)]}o', alpha=0.5)
    axs.set_ylabel('Precision')
    axs.set_xlabel('Recall')
    # axs.set_title(split)
    axs.legend()
    axs.grid()

    out_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'prec_recall', f'{split}_{out_name}')
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_file)
    if(show):
        plt.show()

def plot_conf_matrix(eval_df_path, split, name='conf_matrix', thresholds=[], min_len=None, sub_dir="", show_annotations=True, show=False):
    '''
    Calculate and plot confusion matrix across all meetings per parameter set
    You can specify thresholds(several) and min_len(one) which you want to include
    If nothing passed, all thresholds and min_lens will be plotted
    '''
    if not show:
        plt.clf() # clear existing plots

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

    hm = sns.heatmap(conf_ratio, yticklabels=sum_vals['threshold'], annot=show_annotations, cmap="YlGnBu")
    hm.set_yticklabels(sum_vals['threshold'], size = 11)
    hm.set_xticklabels(labels, size = 12)
    plt.ylabel('threshold', size=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'conf_matrix', f'{name}.png')
    Path(plot_file).parent.mkdir(exist_ok=True, parents=True)
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

def compare_prec_recall(dirs_with_labels, min_len, thresholds, split, sub_dir="", show=False):
    '''
    Compare the prec-recall curve of different experiments
    The input should be a list of tuples containing elements of (dir-path, label).
    Each dir-path needs to be a directory ending in 'pred' containing the predictions for each split 
    in a subfolder 'train', 'dev' and 'test'
    - min_len: for which min_len parameter should the prec-recall curve be plotted
    - thresholds: only plot for those thresholds
    '''
    dfs = []
    for (dir, label) in dirs_with_labels:
        df = pd.read_csv(f"{dir}/{split}_{cfg.ANALYSIS['sum_stats_cache_file']}")
        df = df[df.min_len == min_len]
        df = df[df.threshold.isin(thresholds)]
        dfs.append((df, label)) 
    plot_prec_recall_curve(dfs, out_name='compare_class_balance_dev_set', sub_dir=sub_dir, split=split,  show=show)

def visualise_experiment(dirs, labels, exp_name, conf_thrs, prec_rec_thrs):
    '''
    Creates three visualisations:
        - conf matrix (one plot each)
        - train-metrics (one plot each)
        - prec-recall curve (all in one plot)

    Args:
        - conf_thrs: Threshold for which to plot confusion matrix
        - prec_rec_thrs: Threshold for which to plot precision and recall 
    '''
    
    dirs_with_labels = list(zip(dirs, labels))

    # Create separate plots for each setting: 1) confusion matrix; 2) train metrics
    for dir, label in dirs_with_labels: 
        plot_conf_matrix(dir, split='dev', name=label, thresholds=conf_thrs, min_len=0.2, sub_dir=exp_name,show_annotations=True, show=False)
    
    # NOTE: If put in one loop with confusion matrix the annotation size in conf-matrix changes
    for dir, label in dirs_with_labels: 
        plot_train_metrics(Path(dir).parent, name=label, sub_dir=exp_name, show=False)
    
    # Create one plot comparing the prec-recall-curve of all settings
    compare_prec_recall(dirs_with_labels, min_len=0.2, thresholds=prec_rec_thrs, sub_dir=exp_name, split='dev', show=False) 

def main():
    all_thrs = np.linspace(0,1,21).round(2)
    four_thrs = [0.2,0.4,0.6,0.8]

    ##################################################
    # Init eval on whole ICSI corpus 
    ##################################################
    # plot_conf_matrix('./results/init_eval_2021/preds/', split='all', name='init_eval', thresholds=[0.2,0.4,0.6,0.8], min_len=0.2, show_annotations=True, show=False)

    #################################################
    # Exp1 - Random selection of non-laughter segments
    #################################################
    dirs = [
        './results/1_to_20_16_03/27000_batches/preds',
        './results/1_to_10_16_03/preds',
        './results/1_to_1_21_03/5000_batches/preds',
        './results/init_eval_2021/preds/' # used as baseline
     ]
    labels= ['1-to-20', '1-to-10', '1-to-1','baseline-gillick']
    visualise_experiment(dirs, labels, 'exp_1', conf_thrs=four_thrs, prec_rec_thrs=all_thrs)

    ##################################################
    #Exp2 - Structured selection of non-laughter segments 
    ##################################################
    dirs = [
        './results/1_to_20_16_03/27000_batches/preds',
        './results/1_to_20_struc_22_03/preds',
        './results/1_to_20_struc_70_silence_22_03/preds',
        './results/1_to_20_struc_70_sil_10_nois/preds'
    ]
    labels = [
        'baseline-1-to-20',  # used as baseline
        'model-a',
        'model-b',
        'model-c'
    ]
    visualise_experiment(dirs, labels, 'exp_2', conf_thrs=four_thrs, prec_rec_thrs=all_thrs)

if __name__ == '__main__':
    main()