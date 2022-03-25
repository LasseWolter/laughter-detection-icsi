import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ANALYSIS as cfg
import analysis.utils as utils
import portion as P
import pickle
import os
from analysis.transcript_parsing import parse


def seg_invalid(row):
    """
    This functions specifies what makes a segment invalid
    Input: row defining an audio segment with the following columns:
        - ['meeting_id', 'part_id', 'chan', 'start', 'end', 'length', 'type', 'laugh_type']
    """
    # If the length is shorter than min_length passed to detection algorithm, mark invalid
    #   - empirically tested -> this doesn't apply to many segments
    return (
        row["length"] < cfg["model"]["min_length"]
        or row["laugh_type"] == "breath-laugh"
    )


def append_to_index(index, row, meeting_id, part_id):
    """
    Append the segment as time-interval (Portion) to the passed index
    The segment is defined by the passed dataframe-row
    """
    start = utils.to_frames(row["start"])
    end = utils.to_frames(row["end"])

    seg_as_interval = P.openclosed(start,end)
    # Append to existing intervals or create new dict entry
    if part_id in index[meeting_id].keys():
        index[meeting_id][part_id] = index[meeting_id][part_id] | seg_as_interval 
    else:
        index[meeting_id][part_id] = seg_as_interval 
    
    seg_len = utils.to_sec(utils.p_len(seg_as_interval))

    index[meeting_id]["tot_len"] += seg_len
    index[meeting_id]["tot_events"] += 1
    return index


def create_laugh_index(df, invalid_index):
    """
    Creates a laugh_index with all transcribed laughter events per particpant per meeting
    Invalid index needs to be passed because invalid laughter segments will be added to it
    The segments are stored as disjunction of closed intervals (using portion library)
    dict structure:
    {
        meeting_id: {
            tot_len: INT,
            tot_events: INT,
            part_id: P.openclosed(start,end) | P.openclosed(start,end),
            part_id: P.openclosed(start,end)| P.openclosed(start,end)
        }
        ...
    }
    """
    laugh_index = {}
    meeting_groups = df.groupby(["meeting_id"])

    for meeting_id, meeting_df in meeting_groups:
        laugh_index[meeting_id] = {}
        laugh_index[meeting_id]["tot_len"] = 0
        laugh_index[meeting_id]["tot_events"] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values("start").groupby(["part_id"])
        for part_id, part_df in part_groups:
            laugh_index[meeting_id][part_id] = P.empty()
            for _, row in part_df.iterrows():
                # If segment is invalid, append to invalid segments index
                if seg_invalid(row):
                    invalid_index = append_to_index(
                        invalid_index, row, meeting_id, part_id
                    )
                    continue

                # If segment is valid, append to laugh segments index
                laugh_index = append_to_index(laugh_index, row, meeting_id, part_id)

    return laugh_index


def create_index_from_df(df):
    """
    Creates an index with all segments defined by the passed dataframe
    The segments are stored as disjunction of closed intervals (using portion library) per participant per meeting
    dict structure (same as laugh_index):
    {
        meeting_id: {
            tot_len: INT,
            tot_events: INT,
            part_id: P.openclosed(start,end) | P.openclosed(start,end),
            part_id: P.openclosed(start,end) | P.openclosed(start,end)
            ...
        }
        ...
    }
    """
    index = {}
    meeting_groups = df.groupby(["meeting_id"])
    for meeting_id, meeting_df in meeting_groups:
        index[meeting_id] = {}
        index[meeting_id]["tot_len"] = 0
        index[meeting_id]["tot_events"] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values("start").groupby(["part_id"])
        for part_id, part_df in part_groups:
            for _, row in part_df.iterrows():
                index = append_to_index(index, row, meeting_id, part_id)

    return index


def get_seg_from_index(index, meeting_id, part_id):
    """
    Return index segment for a specific participant of a specific meeting.
    If meeting_id or part_id don't exist in index, return empty interval
    """
    if meeting_id in index.keys():
        return index[meeting_id].get(part_id, P.empty())
    return P.empty()


def create_silence_index(laugh_index, invalid_index, noise_index, speech_index):
    # TODO: Not used at the moment
    """
    Index of those intervals that contain no transcriptions.

    Take whole audio files for each participant for each meeting and subtract all
    transcribed segments
    dict_structure (same as laugh_index - without tot_events)
    {
        meeting_id: {
            tot_len: INT,
            part_id: P.openclosed(start,end) | P.openclosed(start,end),
            part_id: P.openclosed(start,end) | P.openclosed(start,end)
        }
        ...
    }
    """
    silence_index = {}
    for _, row in parse.info_df.iterrows():
        if row.meeting_id not in silence_index.keys():
            silence_index[row.meeting_id] = {}

        end_frame = utils.to_frames(row.length)
        full_interval = P.openclosed(0, end_frame)
        silence_seg = (
            full_interval
            - get_seg_from_index(laugh_index, row.meeting_id, row.part_id)
            - get_seg_from_index(invalid_index, row.meeting_id, row.part_id)
            - get_seg_from_index(speech_index, row.meeting_id, row.part_id)
            - get_seg_from_index(noise_index, row.meeting_id, row.part_id)
        )
        silence_index[row.meeting_id][row.part_id] = silence_seg
        silence_index[row.meeting_id]["tot_length"] = utils.to_sec(utils.p_len(silence_seg))

    return silence_index


#############################################
# EXECUTED ON IMPORT
#############################################
'''
Load from disk if possible. o/w create indices from scratch
'''

cache_file = ".cache/preprocessed_indices.pkl"
force_recompute = cfg['force_index_recompute']

if not force_recompute and os.path.isfile(cache_file):
    print('==========================\nLOADING INDICES FROM DISK\nTo recompute set `force_index_recompute=True` in config.py\n')
    with open(cache_file, "rb") as f:
        mega_index = pickle.load(f)
    invalid_index = mega_index['invalid']
    laugh_index = mega_index['laugh']
    noise_index = mega_index['noise']
    speech_index = mega_index['speech']
    silence_index = mega_index['silence']
else:
    print('Creating indices from transcripts...')
    print('(this can take a while)')
    # The following indices are dicts that contain segments of a particular type per participant per meeting
    invalid_index = create_index_from_df(parse.invalid_df)
    laugh_index = create_laugh_index(parse.laugh_only_df, invalid_index=invalid_index)
    speech_index = create_index_from_df(parse.speech_df)
    noise_index = create_index_from_df(parse.noise_df)

    silence_index = create_silence_index(
        laugh_index, invalid_index, noise_index, speech_index
    )

    mega_index = {
        "invalid": invalid_index,
        "laugh": laugh_index,
        "speech": speech_index,
        "noise": noise_index,
        "silence": silence_index,
    }

    # Create .cache dir if it doesn't exist
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(mega_index, f)
