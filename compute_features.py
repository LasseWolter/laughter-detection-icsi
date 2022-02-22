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
from tqdm import tqdm

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

def compute_features_per_split(icsi_manifest, output_dir, num_jobs=8, use_kaldi=False):
    '''
    Create a cutset for each split, containing a feature representation for each channel over the 
    whole duration of each meeting in that split.
    The feature representation of each channel can then be used to create smaller cuts from it.
    '''
    feats_dir = os.path.join(output_dir, 'feats')
    cutset_dir = os.path.join(output_dir, 'cutsets')

    subprocess.run(['mkdir', '-p', cutset_dir])

    # Using 100frams and 40bins -> ASR standard
    extrac = Fbank(FbankConfig(num_filters=40))
    if (use_kaldi):
            print('Using Kaldifeat-Extractor...')
            extrac = KaldifeatFbank(KaldifeatFbankConfig(
                mel_opts=KaldifeatMelOptions(num_bins=40)))
    
    for split in SPLITS:
        # One cutset representing all meetings (with all their channels) from that split
        split_cutset = CutSet.from_manifests(recordings=icsi_manifest[split]['recordings'],
                                            supervisions=icsi_manifest[split]['supervisions'])

        # Prevents possible error: 
        # See: https://github.com/lhotse-speech/lhotse/issues/559
        torch.set_num_threads(1)

        if use_kaldi:
            split_feat_cuts = split_cutset.compute_and_store_features_batch(
                extractor=extrac,
                storage_path=feats_dir,
                num_workers=num_jobs
            )
        else:
            split_feat_cuts = split_cutset.compute_and_store_features(
                extractor=extrac,
                storage_path=feats_dir,
                num_jobs=num_jobs
            )
        
        split_feat_cuts.to_jsonl(os.path.join(cutset_dir, f'{split}_feats.jsonl'))


def compute_features_for_cuts(icsi_manifest, data_dfs_dir, output_dir, split_feats_dir, num_jobs=8, min_seg_duration=1.0, use_kaldi=False, force_feature_recompute=False):
    '''
    Takes an icsi manifest and a directory containing dataframes which define cuts for the different splits.
    Creates cutsets for each split representing the cuts defined by the dataframes.
    The dataframes define one cut per row in the following format:
        [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
    '''

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
    # + Read the cutsets for each split containing the feature representation for the whole tracks
    dfs = {}
    split_feat_cutsets = {} 
    for split in SPLITS:
        dfs[split] = pd.read_csv(os.path.join(
            data_dfs_dir, f'{split}_df.csv'))

        split_feat_cutsets[split] = CutSet.from_jsonl(os.path.join(split_feats_dir, 'cutsets', f'{split}_feats.jsonl'))

    # We use the existing dataframe to create a corresponding cut for each row
    # Supervisions stating laugh/non-laugh are attached to each cut
    # No audio data is actually loaded into memory or stored to disk at this point.
    # Columns of dataframes look like this:
    #   cols = ['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']

    for split, df in dfs.items():
        cut_list = []
        for ind, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Creating features for {split}-split:'):
            meeting_id = row.meeting_id
            chan_name = row.chan_id
            chan_id = chan_idx_map[meeting_id][chan_name]
            # Get track for this particular channel in this meeting
            # [0] because we know that only one meeting will match this query
            row_track = split_feat_cutsets[split].filter(lambda c: (c.id.startswith(meeting_id) and c.channel ==chan_id))[0]

            # Get a cut that represents the subsample from this track 
            row_cut = row_track.truncate(offset=row.sub_start, duration=row.sub_duration).pad(duration=min_seg_duration, preserve_id=True)
            # rec = icsi_manifest[split]['recordings'][meeting_id]
            # # Create supervision segment indicating laughter or non-laughter by passing a
            # # dict to the custom field -> {'is_laugh': 0/1}
            # # TODO: change duration from hardcoded to a value from a config file
            sup = SupervisionSegment(id=f'sup_{row_cut.id}', recording_id=row_cut.recording.id, start=row.sub_start,
                                     duration=min_seg_duration, channel=chan_id, custom={'is_laugh': row.label})

            # We padded to the right, so [0] will be the MonoCut to which we want to add the supervision
            # This supervision will then also be the supervision for the MixedCut  
            row_cut.tracks[0].cut.supervisions.append(sup)

            # # Pad cut-subsample to a minimum of 1s
            # # Do this because there are laugh segments that are shorter than 1s
            # cut = MonoCut(id=f'{split}_{ind}', start=row.sub_start, duration=row.sub_duration,
            #               recording=rec, channel=chan_id, supervisions=[sup]).pad(duration=1.0)

            cut_list.append(row_cut)


        # Create the actual cutset for this split
        cuts = CutSet.from_cuts(cut_list)

        # Shuffle cutset for better training. In the data_dfs the rows aren't shuffled.
        # At the top are all speech rows and the bottom all laugh rows
        cuts = cuts.shuffle()
        cuts_with_feats_file = os.path.join(cutset_dir, f'{split}_cutset_with_feats.jsonl')
        cuts.to_jsonl(cuts_with_feats_file)

    # print('Write cutset_dict to disk...')
    # with open(os.path.join(cutset_dir, 'cutset_dict_without_feats.pkl'), 'wb') as f:
    #     pickle.dump(cutset_dict, f)

    # for split, cutset in cutset_dict.items():
    #     print(f'Computing features for {split}...')
    #     # Choose frame_shift value to match the hop_length of Gillick et al
    #     # 0.2275 = 16 000 / 364 -> [frame_rate / hop_length]
    #     extrac = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))

    #     if (use_kaldi):
    #         print('Using Kaldifeat-Extractor...')
    #         extrac = KaldifeatFbank(KaldifeatFbankConfig(
    #             mel_opts=KaldifeatMelOptions(num_bins=128)))

    #     # Prevents possible error: 
    #     # See: https://github.com/lhotse-speech/lhotse/issues/559
    #     torch.set_num_threads(1)

    #     # File in which the CutSet object (which contains feature metadata) was/will be stored
    #     cuts_with_feats_file = os.path.join(
    #         cutset_dir, f'{split}_cutset_with_feats.jsonl')

    #     # If file already exist, load it from disk
    #     if(os.path.isfile(cuts_with_feats_file) and not force_feature_recompute):
    #         print("LOADING FEATURES FROM DISK - NOT RECOMPUTING")
    #         cuts = CutSet.from_jsonl(cuts_with_feats_file)
    #     else:
    #         if (use_kaldi):
    #             cuts = cutset.compute_and_store_features_batch(
    #                 extractor=extrac,
    #                 storage_path=feats_dir,
    #                 num_workers=num_jobs
    #             )
    #         else:
    #             cuts = cutset.compute_and_store_features(
    #                 extractor=extrac,
    #                 storage_path=feats_dir,
    #                 num_jobs=num_jobs
    #             )

    #         # If padding is added the returned cut is a MixedCut 
    #         # When features are computed for this MixedCut a MonoCut with no recording attached  
    #         # is returned (see: https://lhotse.readthedocs.io/en/latest/api.html#lhotse.cut.MixedCut.compute_and_store_features)
    #         # This is an issue because a supervision_segment still has the recording_id 
    #         # from the original assignment (which is required for supervision_segment)
    #         # Thus, we re-attach the correct recording and channel to such cuts
    #         # Need to do the same thing for the features of such cuts 
    #         # TODO: Find a better way of doing this
    #         for i in range(0, len(cuts)):
    #             if (not cuts[i].has_recording):
    #                 # Get the recording_id from the supervision segment
    #                 meeting_id = cuts[i].supervisions[0].recording_id 
    #                 chan_id = cuts[i].supervisions[0].channel 
    #                 rec = icsi_manifest[split]['recordings'][meeting_id]
    #                 cuts[i].recording = rec
    #                 cuts[i].channel = chan_id
    #                 cuts[i].features.recording_id = rec.id
    #                 cuts[i].features.channels = chan_id

    #         # Shuffle cutset for better training. In the data_dfs the rows aren't shuffled.
    #         # At the top are all speech rows and the bottom all laugh rows
    #         cuts = cuts.shuffle()
    #         cuts.to_jsonl(cuts_with_feats_file)
            
        
def main(env_file='.env'):
    ''' 
    Creates manifest and computes features for configs specified in .env file passed to this function
    '''
    dotenv.load_dotenv(env_file)
    audio_dir = os.getenv('AUDIO_DIR')
    transcript_dir = os.getenv('TRANSCRIPT_DIR')
    manifest_dir = os.getenv('MANIFEST_DIR')
    output_dir = os.getenv('OUTPUT_DIR')
    split_feat_dir = os.getenv('SPLIT_FEAT_DIR')
    data_dfs_dir = os.getenv('DATA_DFS_DIR')
    num_jobs = int(os.getenv('NUM_JOBS')) if os.getenv('NUM_JOBS') else 8
    min_seg_duration = float(os.getenv('MIN_SEG_DURATION')) if os.getenv('MIN_SEG_DURATION') else 1.0
    use_kaldi = os.getenv('USE_KALDI') == 'True' if os.getenv('USE_KALDI') else False

    icsi_manifest = create_manifest(audio_dir, transcript_dir, manifest_dir)
    compute_features_for_cuts(icsi_manifest=icsi_manifest, 
                     data_dfs_dir=data_dfs_dir,
                     output_dir=output_dir,
                     split_feats_dir=split_feat_dir,
                     num_jobs=num_jobs,
                     min_seg_duration=min_seg_duration, 
                     use_kaldi=use_kaldi)

    # compute_features_per_split(icsi_manifest=icsi_manifest, 
    #                  output_dir=split_feat_dir,
    #                  num_jobs=num_jobs,
    #                  use_kaldi=use_kaldi)


if __name__ == "__main__":
    main()
