from analysis.transcript_parsing import parse
import numpy as np
import analysis.preprocess as prep
import analysis.utils as utils
import portion as P
from config import ANALYSIS as cfg
import pandas as pd
import os
import subprocess
from dotenv import load_dotenv, find_dotenv

# Taken from lhotse icsi recipe to minimise speaker overlap
PARTITIONS = {
    'train': [
        "Bdb001", "Bed002", "Bed003", "Bed004", "Bed005", "Bed006", "Bed008", "Bed009",
        "Bed010", "Bed011", "Bed012", "Bed013", "Bed014", "Bed015", "Bed016", "Bed017",
        "Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007", "Bmr008", "Bmr009",
        "Bmr010", "Bmr011", "Bmr012", "Bmr014", "Bmr015", "Bmr016", "Bmr019", "Bmr020",
        "Bmr022", "Bmr023", "Bmr024", "Bmr025", "Bmr026", "Bmr027", "Bmr028", "Bmr029",
        "Bmr030", "Bmr031", "Bns002", "Bns003", "Bro003", "Bro004", "Bro005", "Bro007",
        "Bro008", "Bro010", "Bro011", "Bro012", "Bro013", "Bro014", "Bro015", "Bro016",
        "Bro017", "Bro018", "Bro019", "Bro022", "Bro023", "Bro024", "Bro025", "Bro026",
        "Bro027", "Bro028", "Bsr001", "Btr001", "Btr002", "Buw001",
    ],
    'dev': ["Bmr021", "Bns001"],
    'test': ["Bmr013", "Bmr018", "Bro021"]
}


def get_random_speech_segment(duration, meeting_id):
    '''
    Get a random speech segment from any channel in the passed meeting
    If there is an overlap between this segment and laughter/invalid regions, resample
    '''
    # Don't create speech samples shorter than the sample duration because these would be padded 
    # during feature computation. Doing this for laugh segments makes sense but for speech segments it's better
    # to use longer segments from the start to avoid having lots of silence in both speech and laughter segments
    duration = max(duration, cfg['train']['subsample_duration'])
    # Only consider segments with passed meeting_id
    info_df = parse.info_df[parse.info_df.meeting_id == meeting_id]
    # Get segment info for this segment from info_df
    num_of_rows = info_df.shape[0]
    row_ind = np.random.randint(0, num_of_rows)
    sample_seg = info_df.iloc[row_ind]
    start = np.random.uniform(0, sample_seg.length-duration)
    speech_seg = P.closed(utils.to_frames(
        start), utils.to_frames(start+duration))

    # If segment overlaps with any laughter or invalid segment, resample
    if (utils.seg_overlaps(speech_seg, [prep.laugh_index, prep.invalid_index], sample_seg.meeting_id, sample_seg.part_id)):
        return get_random_speech_segment(duration, meeting_id)
    else:
        sub_start, sub_duration = get_subsample(
            start, duration, cfg['train']['subsample_duration'])
        return [start, duration, sub_start, sub_duration, sample_seg.path, meeting_id, sample_seg.chan_id, 0]


def get_subsample(start, duration, subsample_duration):
    '''
    Take a segment defined by (start, duration) and return a subsample of passed duration within that region
    '''
    # Taking min is important because otherwise the subsample start can fall out of the range of the laughter
    # and even get negative (e.g. if start=0.1, duration=0.5 -> 0.1+0.5-1.0 = -0.4)
    # We also need to limit the duration to the length of the original segment in case it's at the very end
    # This can lead to segments shorter than 1s. These are padded when features are computed
    sub_dur = min(duration, subsample_duration)
    subsample_start = np.random.uniform(
        start, start+duration - sub_dur)
    return subsample_start, sub_dur


def create_data_df(data_dir, speech_segs_per_laugh_seg):
    '''
    Create 3 dataframes (train,dev,test) with data exactly structured like in the model by Gillick et al.
    Columns:
        [region start, region duration, subsampled region start, subsampled region duration, audio path, label]

    Subsampled region are sampled once during creation. Later either the sampled values can be used or resampling can happen.
    (see Gillick et al. for more details)
    Duration of the subsamples is defined in config.py
    '''
    np.random.seed(cfg['train']['random_seed'])
    speech_seg_lists = {'train': [], 'dev': [], 'test': []}
    laugh_seg_lists = {'train': [], 'dev': [], 'test': []}

    meeting_groups = parse.laugh_only_df.groupby('meeting_id')

    for meeting_id, meeting_laugh_df in meeting_groups:
        split = 'train'
        if meeting_id in PARTITIONS['dev']:
            split = 'dev'
        elif meeting_id in PARTITIONS['test']:
            split = 'test'

        # For each laughter segment get a random speech segment with the same length
        for _, laugh_seg in meeting_laugh_df.iterrows():
            # Get and append random speech segment of same length as current laugh segment
            # Get num of speech segment per one laughter segment defined in config.py
            for _ in range(0, speech_segs_per_laugh_seg):
                speech_seg_lists[split].append(
                    get_random_speech_segment(laugh_seg.length, meeting_id))

            # Subsample laugh segment and append to list
            audio_path = os.path.join(
                laugh_seg.meeting_id, f'{laugh_seg.chan_id}.sph')
            sub_start, sub_duration = get_subsample(
                laugh_seg.start, laugh_seg.length, cfg['train']['subsample_duration'])

            laugh_seg_lists[split].append(
                [laugh_seg.start, laugh_seg.length, sub_start, sub_duration, audio_path, laugh_seg.meeting_id, laugh_seg.chan_id, 1])

    # Columns for data_dfs - same for speech and laughter as they will be combined to one df
    cols = ['start', 'duration', 'sub_start', 'sub_duration', 
            'audio_path', 'meeting_id', 'chan_id', 'label']

    # Create output directory for dataframes
    subprocess.run(['mkdir', '-p', data_dir])

    for split in PARTITIONS.keys():  # [train,,test]
        speech_df = pd.DataFrame(speech_seg_lists[split], columns=cols)
        laugh_df = pd.DataFrame(laugh_seg_lists[split], columns=cols)
        whole_df = pd.concat([speech_df, laugh_df], ignore_index=True)
        # Round all floats to certain number of decimals (defined in config)
        whole_df = whole_df.round(cfg['train']['float_decimals'])

        # Make sure there are no negative start times or durations
        assert whole_df[whole_df.start <
                        0].shape[0] == 0, "Found row with negative start-time"
        assert whole_df[whole_df.duration <
                        0].shape[0] == 0, "Found row with negative duration"
        assert whole_df[whole_df.sub_start <
                        0].shape[0] == 0, "Found row with negative sub_start-time"
        assert whole_df[whole_df.sub_duration <
                        0].shape[0] == 0, "Found row with negative sub_duration"

        # Check that only valid lables are in the dataframe
        assert whole_df[~whole_df.label.isin(
            [0, 1])].shape[0] == 0, "There are labels which are not 0 or 1"

        # Check that df only contains correct meeting ids for this split
        audio_paths = whole_df.audio_path.unique().tolist()
        meeting_ids = set(map(lambda x: x.split('/')[0], audio_paths))
        mismatched_meetings = meeting_ids - set(PARTITIONS[split])
        assert len(
            mismatched_meetings) == 0, f"Found meetings in {split}_df with meeting_id not corresponding to that split, namely: {mismatched_meetings}"

        # Save file to disk
        whole_df.to_csv(os.path.join(data_dir, f'{split}_df.csv'), index=False)


if __name__ == "__main__":
    load_dotenv(find_dotenv('.env'))
    data_dfs_dir = os.getenv('DATA_DFS_DIR')
    speech_segs_per_laugh_seg = int(os.getenv('SPEECH_SEGS_PER_LAUGH_SEG'))
    create_data_df(data_dfs_dir, speech_segs_per_laugh_seg)
