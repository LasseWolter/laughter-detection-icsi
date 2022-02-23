# Example training commands:
# python train.py --config=mlp_mfcc --batch_size=32 --checkpoint_dir=./checkpoints/mlp_baseline_tst
# python train.py --config=resnet_base --batch_size=32 --checkpoint_dir=./checkpoints/resnet_tst
# python train.py --config=resnet_with_augmentation --batch_size=32 --checkpoint_dir=./checkpoints/resnet_aug_tst

# python train.py --config=resnet_with_augmentation --batch_size=32 --checkpoint_dir=./checkpoints/resnet_aug_audioset_tst --train_on_noisy_audioset=True

import json
import load_data
from functools import partial
import configs
import models
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from torch import optim, nn
import os
import sys
import pickle
import time
import argparse
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
# Lhotse imports
from torch.utils.data import DataLoader
from lhotse import CutSet
from lhotse.dataset import VadDataset, SingleCutSampler
from dataclasses import dataclass

sys.path.append('./utils/')
import audio_utils
import torch_utils

warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class MetricEntry():
    accuracy: float
    precision: float
    recall: float
    loss: float
    epoch: int

    def to_list(self): 
        '''
        Returns fields as list in the following order 
        [precision, recall , accuracy, loss]
        '''
        return [self.precision, self.recall, self.accuracy, self.loss]
    
# Stores metrics during training (on train-set and small val-batches) in the following format
# Also stores the global step at which an epoch finished for convenient plotting in a metric-visualisation 
METRICS_DICT = {}
'''
{
    num_batches_processed: {
        "train": MetricsEntry
        "val": MetricsEntry
    }, 
    num_batches_processed: {
    "train": MetricsEntry
    "val": MetricsEntry
    }, 
    ...
}
'''
learning_rate = 0.01  # Learning rate.
decay_rate = 0.9999  # Learning rate decay per minibatch.
min_learning_rate = 0.000001  # Minimum learning rate.

sample_rate = 16000
num_train_steps = 100000

parser = argparse.ArgumentParser()

######## REQUIRED ARGS #########
# Load a preset configuration object. Defines model size, etc. Required
parser.add_argument('--config', type=str, required=True)

# Set a directory to store model checkpoints and tensorboard. Creates a directory if doesn't exist
parser.add_argument('--checkpoint_dir', type=str, required=True)

# Set root data directory containing "Signals/<meeting_id>/<channel>.sph audio files
parser.add_argument('--data_root', type=str, required=True)

######## OPTIONAL ARGS #########
# Number of epochs for which the training should run for
parser.add_argument('--num_epochs', type=int, default=1)

# Directory containing the Lhotse manifest, cutsets and feature representations 
# This should be a relative path from the data_root-dir
parser.add_argument('--lhotse_dir', type=str, default='lhotse')

# Directory containing the Dataframes for train, val, test data
# These dataframes contain the segment information for speech/laughter segments
parser.add_argument('--data_dfs_dir', type=str, default='data_dfs')

# Set batch size. Overrides batch_size set in the config object
parser.add_argument('--batch_size', type=str)

# Default to use GPU. can set to 'cpu' to override
parser.add_argument('--torch_device', type=str, default='cuda')

# Number of processes for parallel processing on cpu. Used mostly for loading in large datafiles
# before training begins or when re-sampling data between epochs
parser.add_argument('--num_workers', type=str, default='8')

# 0.5 unless specified here
parser.add_argument('--dropout_rate', type=str, default='0.5')

# number of batches to accumulate before applying gradients
parser.add_argument('--gradient_accumulation_steps', type=str, default='1')

# include_words flag - if set, data loader will include laughter combined with words
# For example, [laughter - I], [laughter - think], ['laughter -so ']
# This option is not used in the paper
parser.add_argument('--include_words', type=str, default=None)

# Audioset noisy-label training flag
# Flag - if set, train on AudioSet with noisy labels, rather than Switchboard with good labels
parser.add_argument('--train_on_noisy_audioset', type=str, default=None)

args = parser.parse_args()

config = configs.CONFIG_MAP[args.config]
checkpoint_dir = args.checkpoint_dir
data_root = args.data_root
data_dfs_dir = args.data_dfs_dir
lhotse_dir = args.lhotse_dir
batch_size = int(args.batch_size or config['batch_size'])
num_epochs = args.num_epochs
log_frequency = config['log_frequency']
torch_device = args.torch_device
num_workers = int(args.num_workers)
dropout_rate = float(args.dropout_rate)
gradient_accumulation_steps = int(args.gradient_accumulation_steps)
metrics_file = os.path.join(checkpoint_dir, 'metrics.csv')

if args.include_words is not None:
    include_words = True
else:
    include_words = False

if args.train_on_noisy_audioset is not None:
    train_on_noisy_audioset = True
else:
    train_on_noisy_audioset = False

##################################################################
####################  Setup Training Model  ######################
##################################################################
def run_training_loop(n_epochs, model, device, checkpoint_dir,
                      optimizer, iterator, log_frequency=25, val_iterator=None, gradient_clip=1.,
                      verbose=True):

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = run_epoch(model, 'train', device, iterator,
                               checkpoint_dir=checkpoint_dir, optimizer=optimizer,
                               log_frequency=log_frequency, checkpoint_frequency=log_frequency,
                               clip=gradient_clip, val_iterator=val_iterator,
                               verbose=verbose, epoch_num=epoch+1)

        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = torch_utils.epoch_time(
                start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')


def run_epoch(model, mode, device, iterator, checkpoint_dir, epoch_num, optimizer=None, clip=None,
              batches=None, log_frequency=None, checkpoint_frequency=None,
              validate_online=True, val_iterator=None, val_batches=None,
              verbose=True):
    """ args:
            mode: 'train' or 'eval'
    """

    def _eval_for_logging(model, device, val_itr, val_iterator, val_batches_per_log):
        model.eval()
        val_losses = []
        # Collect target and pred values for all batches and calc metrics at the end 
        val_trgs = torch.tensor([])
        val_preds = torch.tensor([]) 

        for j in range(val_batches_per_log):
            try:
                val_batch = val_itr.next()
            except StopIteration:
                val_itr = iter(val_iterator)
                val_batch = val_itr.next()

            val_loss, trgs, preds= _eval_batch(
                model, device, val_batch, return_raw=True)

            val_trgs = torch.cat((val_trgs, trgs))
            val_preds = torch.cat((val_preds, preds))
            val_losses.append(val_loss)

        acc, prec, recall = _calc_metrics(val_trgs, val_preds)
        model.train()
        return val_itr, np.mean(val_losses), acc, prec, recall 

    def _calc_metrics(trgs, preds):
        '''
        Calculates accuracy, precision and recall and returns them in that order
        '''
        acc = torch.sum(preds == trgs).float()/len(trgs)

        # Calculate necessary numbers for prec and recall calculation
        # '==' operator on tensors is applied element-wise
        # '*' exploits the fact that True*True = 1
        corr_pred_laughs = torch.sum((preds == trgs) * (preds == 1)).float()
        total_trg_laughs = torch.sum(trgs == 1).float()
        total_pred_laughs = torch.sum(preds == 1).float()

        if total_pred_laughs == 0:
            prec = torch.tensor(1.0) 
        else:
            prec = corr_pred_laughs/total_pred_laughs

        recall = corr_pred_laughs/total_trg_laughs

        # Returns only the content of the torch tensor
        return acc.item(), prec.item(), recall.item()

    def _eval_batch(model, device, batch, batch_index=None, clip=None, return_raw=False):
        '''
        Evaluates one batch
            'return_raw'=True: allows returning the raw target and prediction values to accumulate them 
            before calculating any metrics
        '''
        if batch is None:
            print("None Batch")
            return 0.

        with torch.no_grad():
            #seqs, labs = batch
            segs = batch['inputs']
            labs = batch['is_laugh']

            src = torch.from_numpy(np.array(segs)).float().to(device)
            src = src[:, None, :, :]  # add additional dimension

            trgs = torch.from_numpy(np.array(labs)).float().to(device)
            output = model(src).squeeze()

            criterion = nn.BCELoss()
            bce_loss = criterion(output, trgs)
            preds = torch.round(output)
            # sum(preds==trg).float()/len(preds)

            # Allows to evaluate several batches together for logging
            # Used to avoid lots of precision=1 because no predictions were made
            if return_raw:
                return bce_loss, trgs, preds

            acc, prec, recall = _calc_metrics(trgs, preds)

            return bce_loss.item(), acc, prec, recall

    def _train_batch(model, device, batch, batch_index=None, clip=None):

        if batch is None:
            print("None Batch")
            return 0.

        #seqs, labs = batch
        segs = batch['inputs']
        labs = batch['is_laugh']

        src = torch.from_numpy(np.array(segs)).float().to(device)
        src = src[:, None, :, :]  # add additional dimension
        trgs = torch.from_numpy(np.array(labs)).float().to(device)

        # optimizer.zero_grad()

        output = model(src).squeeze()

        criterion = nn.BCELoss()

        preds = torch.round(output)

        acc, prec, recall = _calc_metrics(trgs, preds)        

        bce_loss = criterion(output, trgs)

        loss = bce_loss
        loss = loss/gradient_accumulation_steps
        loss.backward()

        if model.global_step % gradient_accumulation_steps == 0:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.zero_grad()

        return bce_loss.item(), acc, prec, recall

    if mode.lower() not in ['train', 'eval']:
        raise Exception("`mode` must be 'train' or 'eval'")

    if mode.lower() == 'train' and validate_online:
        #val_batches_per_epoch = torch_utils.num_batches_per_epoch(val_iterator)
        #val_batches_per_log = int(np.round(val_batches_per_epoch))
        val_batches_per_log = 10  # TODO hardcoded for now
        val_itr = iter(val_iterator)

    if mode == 'train':
        if optimizer is None:
            raise Exception("Must pass Optimizer in train mode")
        model.train()
        _run_batch = _train_batch
    elif mode == 'eval':
        model.eval()
        _run_batch = _eval_batch

    epoch_loss = 0

    optimizer = optim.Adam(model.parameters())

    if iterator is not None:
        batch_losses = []
        batch_accs = []
        batch_precs = []
        batch_recalls = []

        num_batches = 0
        for i, batch in tqdm(enumerate(iterator)):
            # learning rate scheduling
            lr = (learning_rate - min_learning_rate) * \
                decay_rate**(float(model.global_step))+min_learning_rate
            optimizer.lr = lr

            batch_loss, batch_acc, batch_prec, batch_recall = _run_batch(model, device, batch,
                                                                         batch_index=i, clip=clip)

            epoch_loss += batch_loss
            model.global_step += 1
            num_batches = +1

            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
            batch_precs.append(batch_prec)
            batch_recalls.append(batch_recall)

            if log_frequency is not None and (model.global_step + 1) % log_frequency == 0:
                # TODO: possibly remove val_itr from return values?
                val_itr, val_loss_at_step, val_acc_at_step, val_prec_at_step, val_recall_at_step = _eval_for_logging(model, device,
                                                                                                                     val_itr, val_iterator, val_batches_per_log)

                is_best = (val_loss_at_step < model.best_val_loss)
                if is_best:
                    model.best_val_loss = val_loss_at_step

                # Init metrics entry for this batch_number (i.e. global step)
                METRICS_DICT[model.global_step] = {}
                # Save metrics for the validation above
                val_metrics = MetricEntry(
                    accuracy= val_acc_at_step,
                    precision= val_prec_at_step,
                    recall= val_recall_at_step,
                    loss= val_loss_at_step,
                    epoch=epoch_num
                )
                METRICS_DICT[model.global_step]['val']= val_metrics

                # Save metrics on training set up to now
                train_metrics = MetricEntry(
                    accuracy=np.mean(batch_accs),
                    precision=np.mean(batch_precs),
                    # Ignore nan values for recall mean calculation
                    recall=np.nanmean(batch_recalls),
                    loss=np.mean(batch_losses),
                    epoch=epoch_num
                )

                # Reset training metrics 
                batch_losses = []
                batch_accs = [] 
                batch_recalls = []
                batch_precs = []

                METRICS_DICT[model.global_step]['train'] = train_metrics

                if verbose:
                    print("\nLogging at step: ", model.global_step)
                    print("Train metrics: ", train_metrics)
                    print("Validation metrics: ", val_metrics)


            if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
                state = torch_utils.make_state_dict(model, optimizer, model.epoch,
                                                    model.global_step, model.best_val_loss)
                torch_utils.save_checkpoint(
                    state, is_best=is_best, checkpoint=checkpoint_dir)

        model.epoch += 1
        return epoch_loss / num_batches


print("Initializing model...")
device = torch.device(torch_device if torch.cuda.is_available() else 'cpu')
print("Using device", device)
model = config['model'](dropout_rate=dropout_rate,
                        linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)
torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)
optimizer = optim.Adam(model.parameters())

if os.path.exists(checkpoint_dir) and os.path.isfile(os.path.join(checkpoint_dir, 'last.pth.tar')):
    torch_utils.load_checkpoint(
        checkpoint_dir+'/last.pth.tar', model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    print("Beginning training...")

def get_audios_from_text_data(data_file_or_lines, h, sr=sample_rate):
    # This function doesn't use the subsampled offset and duration
    # So it will need to be handled later, in the data loader
    #column_names = ['offset','duration','audio_path','label']
    column_names = ['offset', 'duration', 'subsampled_offset',
                    'subsampled_duration', 'audio_path', 'label']
    audios = []
    if type(data_file_or_lines) == type([]):
        df = pd.DataFrame(data=data_file_or_lines, columns=column_names)
    else:
        df = pd.read_csv(data_file_or_lines, sep='\t',
                         header=None, names=column_names)

    audio_paths = list(df.audio_path)
    offsets = list(df.offset)
    durations = list(df.duration)
    for i in tqdm(range(len(audio_paths))):
        aud = h[audio_paths[i]][int(offsets[i]*sr)
                                    :int((offsets[i]+durations[i])*sr)]
        audios.append(aud)
    return audios


def time_dataloading(iterations, dataloader, is_lhotse=False):
    '''
    Evaluate the time it takes to load data from the dataloader
    The number of iterations means how many batches will be fetched from the dataloader
    'is_lhotse' states if this is an lhotse dataloader whose batch structure is slightly different
    '''
    start_time = time.time()
    num_of_its = iterations
    for i in range(0, num_of_its):
        if is_lhotse:
            batch = next(iter(dataloader))
            sigs = batch['inputs']
            labels = batch['is_laugh']
        else:
            sigs, labels = next(iter(dataloader))

        print(f'Signal batch shape: {sigs.shape[0]}')
        if is_lhotse:
            print(f"Lables batch shape: {labels.shape[0]}")
        else:
            print(f"Lables batch shape: {len(labels)}")

        print(f"Label of first signal in batch: {labels[0]}")

    exec_time = time.time() - start_time
    print(f'num_of_workers: {num_workers}')
    print(f'Execution took for {num_of_its} batches: {exec_time}s')
    print(
        f'Average time per batch (size: {batch_size}): {exec_time/float(num_of_its)}')

def update_metrics_on_disk():
    metric_rows = []
    for batch_num, entry_dict in METRICS_DICT.items():
        train_entry = entry_dict['train'].to_list()
        val_entry = entry_dict['val'].to_list()
        # Epoch will be the same for training and validation - just take value from training MetricsEntry-object
        metric_rows.append([batch_num, entry_dict['train'].epoch] + train_entry + val_entry)
    
    cols = ['batch_num', 'epoch', 'train_prec', 'train_rec', 'train_acc', 'train_loss', 'val_prec', 'val_rec', 'val_acc', 'val_loss'] 
    metrics_df = pd.DataFrame(metric_rows, columns=cols)

    # Concat with existing metrics if they exist
    if os.path.isfile(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_df, metrics_df]) 

    metrics_df.to_csv(metrics_file, index=False)

print("Preparing training set...")

cutset_dir = os.path.join(data_root, lhotse_dir, 'cutsets')

dev_loader = load_data.create_training_dataloader(cutset_dir, 'dev')

train_loader = load_data.create_training_dataloader(cutset_dir, 'dev')
# time_dataloading(1, lhotse_loader, is_lhotse=True)


start_time = time.time()
run_training_loop(n_epochs=num_epochs, model=model, device=device,
                  iterator=train_loader, checkpoint_dir=checkpoint_dir, optimizer=optimizer,
                  log_frequency=log_frequency, val_iterator=dev_loader,
                  verbose=True)

tot_train_time = time.time() - start_time
time_in_m = tot_train_time/60
time_in_h = tot_train_time/3600

time_per_epoch = tot_train_time/num_epochs
epoch_time_in_m = time_per_epoch/60
epoch_time_in_h = time_per_epoch/3600
print("Ran {num_epochs} epochs.")
print(
    f"Total training time[in three different formats s/min/h]:\n{tot_train_time:.2f}s\n{time_in_m:.2f}m\n{time_in_h:.2f}h")
print('---------------')
print(
    f"Time per epoch time[in three different formats s/min/h]:\n{time_per_epoch:.2f}s\n{epoch_time_in_m:.2f}m\n{epoch_time_in_h:.2f}h")

update_metrics_on_disk()
