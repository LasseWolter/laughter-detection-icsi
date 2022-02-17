'''
REQUIREMENTS: 
    1. `chan_idx_map.pkl` file next to the data_frames in the dataframe folder
    2. .env file with configs (can be passed to main function) 

This script creates feature representations for audio segments present in a given dataframe
The dataframe must have the following columns: 
['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']
with the following meaning  
[region start, region duration, subsampled region start, subsampled region duration, audio path, label]

EXAMPLE .env file: 
    AUDIO_DIR=./data/icsi/speech
    TRANSCRIPT_DIR=./data/icsi/
    DATA_DFS_DIR=./data/icsi/data_dfs
    OUTPUT_DIR=test_output
    NUM_JOBS=8

USAGE: python compute_features
'''
import torch
from lhotse import CutSet, Fbank, FbankConfig, MonoCut, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.features.kaldifeat import KaldifeatMelOptions
from lhotse.recipes import prepare_icsi
from lhotse import SupervisionSegment, SupervisionSet, RecordingSet
import pandas as pd
import pickle
import os
import subprocess
import dotenv

SPLITS = ['train', 'dev', 'test']


def create_manifest(audio_dir, transcripts_dir, output_dir, force_manifest_reload=False):
    '''
    Create or load lhotse manifest for icsi dataset.  
    If it exists on disk, load it. Otherwise create it using the icsi_recipe
    '''
    # Prepare data manifests from a raw corpus distribution.
    # The RecordingSet describes the metadata about audio recordings;
    # the sampling rate, number of channels, duration, etc.
    # The SupervisionSet describes metadata about supervision segments:
    # the transcript, speaker, language, and so on.
    if(os.path.isdir(output_dir) and not force_manifest_reload):
        print("LOADING MANIFEST DIR FROM DISK - not from raw icsi files")
        icsi = {}
        for split in SPLITS:
            icsi[split] = {}
            rec_set = RecordingSet.from_jsonl(os.path.join(
                output_dir, f'recordings_{split}.jsonl'))
            sup_set = SupervisionSet.from_jsonl(os.path.join(
                output_dir, f'supervisions_{split}.jsonl'))
            icsi[split]['recordings'] = rec_set
            icsi[split]['supervisions'] = sup_set
    else:
        icsi = prepare_icsi(
            audio_dir=audio_dir, transcripts_dir=transcripts_dir, output_dir=output_dir)

    return icsi


def compute_features(icsi_manifest, data_dfs_dir, output_dir, num_jobs=8, use_kaldi=False, force_feature_recompute=False):

    feats_dir = os.path.join(output_dir, 'feats')
    cutset_dir = os.path.join(output_dir, 'cutsets')
    # Create directory for storing lhotse cutsets
    # Feats dir is automatically created by lhotse
    subprocess.run(['mkdir', '-p', cutset_dir])

    # Load the channel to id mapping from disk
    # If this changed at some point (which it shouldn't) this file would have to
    # be recreated
    # TODO: find a cleaner way to implement this
    chan_map_file = open(os.path.join(data_dfs_dir, 'chan_idx_map.pkl'), 'rb')
    chan_idx_map = pickle.load(chan_map_file)

    # Read data_dfs containing the samples for train,val,test split
    dfs = {}
    for split in SPLITS:
        dfs[split] = pd.read_csv(os.path.join(
            data_dfs_dir, f'{split}_df.csv'))

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
            rec = icsi_manifest[split]['recordings'][meeting_id]
            # Create supervision segment indicating laughter or non-laughter by passing a
            # dict to the custom field -> {'is_laugh': 0/1}
            # TODO: change duration from hardcoded to a value from a config file
            sup = SupervisionSegment(id=f'sup_{split}_{ind}', recording_id=rec.id, start=row.sub_start,
                                     duration=1.0, channel=chan_id, custom={'is_laugh': row.label})

            # Pad cut-subsample to a minimum of 1s
            # Do this because there are laugh segments that are shorter than 1s
            cut = MonoCut(id=f'{split}_{ind}', start=row.sub_start, duration=row.sub_duration,
                          recording=rec, channel=chan_id, supervisions=[sup]).pad(duration=1.0)
            cut_list.append(cut)

        cutset_dict[split] = CutSet.from_cuts(cut_list)

    print('Write cutset_dict to disk...')
    with open(os.path.join(cutset_dir, 'cutset_dict_without_feats.pkl'), 'wb') as f:
        pickle.dump(cutset_dict, f)

    for split, cutset in cutset_dict.items():
        print(f'Computing features for {split}...')
        # Choose frame_shift value to match the hop_length of Gillick et al
        # 0.2275 = 16 000 / 364 -> [frame_rate / hop_length]
        fbank_conf = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))

        if (use_kaldi):
            print('Using Kaldifeat-Extractor...')
            fbank_conf = KaldifeatFbank(KaldifeatFbankConfig(
                mel_opts=KaldifeatMelOptions(num_bins=128)))

        torch.set_num_threads(1)

        # File in which the CutSet object (which contains feature metadata) was/will be stored
        cuts_with_feats_file = os.path.join(
            cutset_dir, f'{split}_cutset_with_feats.jsonl')

        # If file already exist, load it from disk
        if(os.path.isfile(cuts_with_feats_file) and not force_feature_recompute):
            print("LOADING FEATURES FROM DISK - NOT RECOMPUTING")
            cuts = CutSet.from_jsonl(f'{split}_cutset_with_feats.jsonl')
        else:
            if (use_kaldi):
                cuts = cutset.compute_and_store_features_batch(
                    extractor=fbank_conf,
                    storage_path=feats_dir,
                    num_workers=num_jobs
                )
            else:
                cuts = cutset.compute_and_store_features(
                    extractor=fbank_conf,
                    storage_path=feats_dir,
                    num_jobs=num_jobs
                )
            # Shuffle cutset for better training. In the data_dfs the rows aren't shuffled.
            # At the top are all speech rows and the bottom all laugh rows
            cuts = cuts.shuffle()
            cuts.to_jsonl(cuts_with_feats_file)


def main(env_file='.env'):
    ''' 
    Creates manifest and computes features for configs specified in .env file passed to this function
    '''
    dotenv.load_dotenv(env_file)
    audio_dir = os.getenv('AUDIO_DIR')
    transcript_dir = os.getenv('TRANSCRIPT_DIR')
    output_dir = os.getenv('OUTPUT_DIR')
    data_dfs_dir = os.getenv('DATA_DFS_DIR')
    num_jobs = int(os.getenv('NUM_JOBS')) if os.getenv('NUM_JOBS') else 8
    use_kaldi = os.getenv('USE_KALDI') == 'True' if os.getenv('USE_KALDI') else False

    icsi_manifest = create_manifest(audio_dir, transcript_dir, output_dir)
    compute_features(icsi_manifest=icsi_manifest, data_dfs_dir=data_dfs_dir,
                     output_dir=output_dir, num_jobs=num_jobs, use_kaldi=use_kaldi)


if __name__ == "__main__":
    main()
