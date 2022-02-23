import os
import sys
import shutil
import torch
from torch import nn

# Import different progress bar depending on environment
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


##################### INITIALIZATION ##########################

def count_parameters(model):
    counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {counts:,} trainable parameters')


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


##################### TENSOR OPERATIONS #######################


def num_batches_per_epoch(generator):
    return len(generator.dataset)/generator.batch_size


################## CHECKPOINTING ##############################

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
            state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            is_best: (bool) True if it is the best model seen till now
            checkpoint: (string) folder where parameters are to be saved

    Modified from: https://github.com/cs230-stanford/cs230-code-examples/
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
            checkpoint: (string) filename which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim) optional: resume optimizer from checkpoint

    Modified from: https://github.com/cs230-stanford/cs230-code-examples/
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    else:
        print("Loading checkpoint at:", checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.epoch = checkpoint['epoch']

    if 'global_step' in checkpoint:
        model.global_step = checkpoint['global_step'] + 1
        print("Loading checkpoint at step: ", model.global_step)

    if 'best_val_loss' in checkpoint:
        model.best_val_loss = checkpoint['best_val_loss']

    return checkpoint


def make_state_dict(model, optimizer=None, epoch=None, global_step=None,
                    best_val_loss=None):
    return {'epoch': epoch, 'global_step': global_step,
            'best_val_loss': best_val_loss, 'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()
            }


##################### TRAINING METHODS ######################

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs