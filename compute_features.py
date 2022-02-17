'''
REQUIREMENT: `chan_idx_map.pkl` file next to the data_frames in the dataframe folder

This script creates feature representations for audio segments present in a given dataframe
The dataframe must have the following columns: 
['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']
with the following meaning  
[region start, region duration, subsampled region start, subsampled region duration, audio path, label]

EXAMPLE USAGE: 
python load_data.py --audio_dir data/icsi/speech/ --transcript_dir data/icsi/ --data_df_dir data/icsi/data_dfs/ --output_dir test_output --debug True
'''
import argparse
import torch
from lhotse import CutSet, Fbank, FbankConfig, MonoCut
from lhotse.recipes import prepare_icsi
from lhotse import SupervisionSegment, SupervisionSet, RecordingSet
import pandas as pd
import pickle
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()

######## REQUIRED ARGS #########
# Directory containing the audio_files from ICSI corpus
parser.add_argument('--audio_dir', type=str, required=True)

# Directory containing the ICSI transcripts
parser.add_argument('--transcript_dir', type=str, required=True)

# Directory which contains the dataframes describing the audio segments we want to create
# features for
parser.add_argument('--data_df_dir', type=str, required=True)

# Directory which will contain the manifest,cutsets and features
parser.add_argument('--output_dir', type=str, required=True)

######## OPTIONAL ARGS #########
# Set DEBUG variable to true, using dummy data instead of whole audio data
# Will use the output folder set above an create a 'debug' folder inside it for the outputs
parser.add_argument('--debug', type=bool, default=False)

# Allows overwriting already stored manifests
# E.g. when using different audio paths the manifest needs to be recreated
parser.add_argument('--force_manifest_reload', type=bool, default=False)

# Set batch size. Overrides batch_size set in the config object
parser.add_argument('--force_feature_recompute', type=bool, default=False)

# Number of processes for parallel processing on cpu.
parser.add_argument('--num_jobs', type=int, default=8)

args = parser.parse_args()

DEBUG = args.debug
FORCE_MANIFEST_RELOAD = args.force_manifest_reload
# allows overwriting already computed features
FORCE_FEATURE_RECOMPUTE = args.force_feature_recompute

#SPLITS = ['train', 'dev', 'test']
SPLITS = ['dev']

# Initialise directories according to the passed arguments
if DEBUG:
    # output_dir: Directory which will contain manifest, cutset dumps and features
    output_dir = os.path.join(args.output_dir, 'debug')
    print('IN DEBUG MODE - loading small amount of data')
else:
    output_dir = args.output_dir

data_df_dir = args.data_df_dir
audio_dir = args.audio_dir
transcripts_dir = args.transcript_dir
manifest_dir = os.path.join(output_dir, 'manifests')
feats_dir = os.path.join(output_dir, 'feats')
cutset_dir = os.path.join(output_dir, 'cutsets')


def create_manifest(audio_dir, transcripts_dir, manifest_dir):
    '''
    Create or load lhotse manifest for icsi dataset.  
    If it exists on disk, load it. Otherwise create it using the icsi_recipe
    '''
    # Prepare data manifests from a raw corpus distribution.
    # The RecordingSet describes the metadata about audio recordings;
    # the sampling rate, number of channels, duration, etc.
    # The SupervisionSet describes metadata about supervision segments:
    # the transcript, speaker, language, and so on.
    if(os.path.isdir(manifest_dir) and not FORCE_MANIFEST_RELOAD):
        print("LOADING MANIFEST DIR FROM DISK - not from raw icsi files")
        icsi = {'train': {}, 'dev': {}, 'test': {}}
        for split in ['train', 'dev', 'test']:
            rec_set = RecordingSet.from_jsonl(os.path.join(
                manifest_dir, f'recordings_{split}.jsonl'))
            sup_set = SupervisionSet.from_jsonl(os.path.join(
                manifest_dir, f'supervisions_{split}.jsonl'))
            icsi[split]['recordings'] = rec_set
            icsi[split]['supervisions'] = sup_set
    else:
        icsi = prepare_icsi(
            audio_dir=audio_dir, transcripts_dir=transcripts_dir, output_dir=manifest_dir)

    return icsi


def compute_features():
    # Create directory for storing lhotse cutsets
    # Manifest dir is automatically created by lhotse's icsi recipe if it doesn't exist
    subprocess.run(['mkdir', '-p', cutset_dir])

    icsi = create_manifest(audio_dir, transcripts_dir, manifest_dir)

    # Load the channel to id mapping from disk
    # If this changed at some point (which it shouldn't) this file would have to
    # be recreated
    # TODO: find a cleaner way to implement this
    chan_map_file = open(os.path.join(data_df_dir, 'chan_idx_map.pkl'), 'rb')
    chan_idx_map = pickle.load(chan_map_file)

    # Read data_dfs containing the samples for train,val,test split
    dfs = {}
    if DEBUG:
        # Dummy data is in the train split
        dfs['train'] = pd.read_csv(os.path.join(
            data_df_dir, f'dummy_df.csv'))
    else:
        for split in SPLITS:
            dfs[split] = pd.read_csv(os.path.join(
                data_df_dir, f'{split}_df.csv'))

    # CutSet is the workhorse of Lhotse, allowing for flexible data manipulation.
    # We use the existing dataframe to create a corresponding cut for each row
    # Supervisions stating laugh/non-laugh are attached to each cut
    # No audio data is actually loaded into memory or stored to disk at this point.
    # Columns of dataframes look like this:
    #   cols = ['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']

    cutset_dict = {}  # will contain CutSets for different splits
    for split, df in dfs.items():
        cut_list = []
        for ind, row in df.iterrows():
            meeting_id = row.audio_path.split('/')[0]
            channel = row.audio_path.split('/')[1].split('.')[0]
            chan_id = chan_idx_map[meeting_id][channel]
            if DEBUG:
                # The meeting used in dummy_df is in the train-split
                rec = icsi['train']['recordings'][meeting_id]
            else:
                # In the icsi recipe the validation split is called 'dev' split
                rec = icsi[split]['recordings'][meeting_id]
            # Create supervision segment indicating laughter or non-laughter by passing a
            # dict to the custom field -> {'is_laugh': 0/1}
            # TODO: change duration from hardcoded to a value from a config file
            sup = SupervisionSegment(id=f'sup_{split}_{ind}', recording_id=rec.id, start=row.sub_start,
                                     duration=1.0, channel=chan_id, custom={'is_laugh': row.label})

            # Pad cut-subsample to a minimum of 1s
            # Do this because there are laugh segments that are shorter than 1s
            cut = MonoCut(id=f'{split}_{ind}', start=row.sub_start, duration=row.sub_duration,
                          recording=rec, channel=chan_id, supervisions=[sup])
            cut_list.append(cut)

        cutset_dict[split] = CutSet.from_cuts(cut_list)

    print('Write cutset_dict to disk...')
    with open(os.path.join(cutset_dir, 'cutset_dict_without_feats.pkl'), 'wb') as f:
        pickle.dump(cutset_dict, f)

    for split, cutset in cutset_dict.items():
        print(f'Computing features for {split}...')
        # Choose frame_shift value to match the hop_length of Gillick et al
        # 0.2275 = 16 000 / 364 -> [frame_rate / hop_length]
        f2 = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))

        torch.set_num_threads(1)

        # File in which the CutSet object (which contains feature metadata) was/will be stored
        cuts_with_feats_file = os.path.join(
            cutset_dir, f'{split}_cutset_with_feats.jsonl')

        # If file already exist, load it from disk
        if(os.path.isfile(cuts_with_feats_file) and not FORCE_FEATURE_RECOMPUTE):
            print("LOADING FEATURES FROM DISK - NOT RECOMPUTING")
            cuts = CutSet.from_jsonl(f'{split}_cutset_with_feats.jsonl')
        else:
            cuts = cutset.compute_and_store_features(
                extractor=f2,
                storage_path=feats_dir,
                num_jobs=args.num_jobs
            )
            # Shuffle cutset for better training. In the data_dfs the rows aren't shuffled.
            # At the top are all speech rows and the bottom all laugh rows
            cuts = cuts.shuffle()
            cuts.to_jsonl(cuts_with_feats_file)


if __name__ == "__main__":
    compute_features()