# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

import configs
import models
import time
from distutils.util import strtobool
from functools import partial
from torch import optim, nn
import laugh_segmenter
import os
import sys
import pickle
import time
import librosa
import argparse
import torch
import numpy as np
import pandas as pd
import scipy.io.wavfile
from tqdm import tqdm
import tgt
import load_data
sys.path.append('./utils/')
import torch_utils
import audio_utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--thresholds', type=str, default='0.5', help='Single value or comma-separated list of thresholds to evaluate')
parser.add_argument('--min_lengths', type=str, default='0.2', help='Single value or comma-separated list of min_lengths to evaluate')
parser.add_argument('--input_audio_file', required=True, type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--save_to_audio_files', type=str, default='True')
parser.add_argument('--save_to_textgrid', type=str, default='False')

args = parser.parse_args()


model_path = args.model_path
config = configs.CONFIG_MAP[args.config]
audio_path = args.input_audio_file
save_to_audio_files = bool(strtobool(args.save_to_audio_files))
save_to_textgrid = bool(strtobool(args.save_to_textgrid))
output_dir = args.output_dir

# Turn comma-separated parameter strings into list of floats 
thresholds = [float(t) for t in args.thresholds.split(',')]
min_lengths = [float(l) for l in args.min_lengths.split(',')]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model

model = config['model'](
    dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)

if os.path.exists(model_path):
    if device == 'cuda':
        torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
    else:
        # Different method needs to be used when using CPU
        # see https://pytorch.org/tutorials/beginner/saving_loading_models.html for details
        checkpoint = torch.load(
            model_path+'/best.pth.tar', lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")

# Load the audio file and features


def load_and_pred(audio_path):
    '''
    Input: audio_path for audio to predict 
    Output: time taken to predict (excluding the generation of output files)
    Loads audio, runs prediction and outputs results according to flag-settings (e.g. TextGrid or Audio)
    '''
    start_time = time.time()  # Start measuring time

    inference_generator = load_data.create_inference_dataloader(audio_path)

    probs = []
    for model_inputs in tqdm(inference_generator):
        # x = torch.from_numpy(model_inputs).float().to(device)
        # Model inputs from new inference generator are tensors already
        model_inputs = model_inputs[:,None,:,:] # add additional dimension
        x = model_inputs.float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs)/float(file_length)

    probs = laugh_segmenter.lowpass(probs)

    # Get a list of instance for each setting passed in  
    instance_dict = laugh_segmenter.get_laughter_instances(
        probs, thresholds=thresholds, min_lengths=min_lengths, fps=fps)

    time_taken = time.time() - start_time  # stop measuring time
    print(f'Completed in: {time_taken:.2f}s')

    for setting, instances in instance_dict.items():
        print(f"Found {len(instances)} laughs for threshold {setting[0]} and min_length {setting[1]}.") 
        instance_output_dir = os.path.join(output_dir, f't_{setting[0]}', f'l_{setting[1]}')
        save_instances(instances, instance_output_dir, save_to_audio_files, save_to_textgrid)

    return time_taken

def save_instances(instances, output_dir, save_to_audio_files, save_to_textgrid):
    '''
    Saves given instances to disk in a form that is specified by the passed parameters. 
    Possible forms:
        1. as audio file
        2. as textgrid file
    '''
    if len(instances) > 0:
        os.system(f"mkdir -p {output_dir}")
        if save_to_audio_files:
            full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
            wav_paths = []
            maxv = np.iinfo(np.int16).max
            if output_dir is None:
                raise Exception(
                    "Need to specify an output directory to save audio files")
            else:
                for index, instance in enumerate(instances):
                    laughs = laugh_segmenter.cut_laughter_segments(
                        [instance], full_res_y, full_res_sr)
                    wav_path = output_dir + "/laugh_" + str(index) + ".wav"
                    scipy.io.wavfile.write(
                        wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                    wav_paths.append(wav_path)
                print(laugh_segmenter.format_outputs(instances, wav_paths))

        if save_to_textgrid:
            laughs = [{'start': i[0], 'end': i[1]} for i in instances]
            tg = tgt.TextGrid()
            laughs_tier = tgt.IntervalTier(name='laughter', objects=[
                tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
            tg.add_tier(laughs_tier)
            fname = os.path.splitext(os.path.basename(audio_path))[0]
            tgt.write_to_file(tg, os.path.join(
                output_dir, fname + '.TextGrid'))

            print('Saved laughter segments in {}'.format(
                os.path.join(output_dir, fname + '_laughter.TextGrid')))

def i_pred():
    """
    Interactive Prediction Shell running until interrupted
    """
    print('Model loaded. Waiting for file input...')
    while True:
        audio_path = input()
        if os.path.isfile(audio_path):
            audio_length = audio_utils.get_audio_length(audio_path)
            print(audio_length)
            load_and_pred(audio_path)
        else:
            print("audio_path doesn't exist. Try again...")


def calc_real_time_factor(audio_path, iterations):
    """
    Calculates realtime factor by reading 'audio_path' and running prediction 'iteration' times 
    """
    if os.path.isfile(audio_path):
        audio_length = audio_utils.get_audio_length(audio_path)
        print(f"Audio Length: {audio_length}")
    else:
        raise ValueError(f"Audio_path doesn't exist. Given path {audio_path}")

    sum_time = 0
    for i in range(0, iterations):
        print(f'On iteration {i+1}')
        sum_time += load_and_pred(audio_path)

    av_time = sum_time/iterations
    # Realtime factor is the 'time taken for prediction' / 'duration of input audio'
    av_real_time_factor = av_time/audio_length
    print(
        f"Average Realtime Factor over {iterations} iterations: {av_real_time_factor:.2f}")

load_and_pred(audio_path)