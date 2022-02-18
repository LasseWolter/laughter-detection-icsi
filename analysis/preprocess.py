import analysis.config as cfg
import analysis.utils as utils
import portion as P
from analysis.transcript_parsing import parse

# The following indices are dicts that contain segments per participant per meeting
laugh_index = {}         # Index containing all laughter segments
invalid_index = {}       # Index containing invalid segments
valid_speech_index = {}  # Index of those intervals containing valid speech


def seg_invalid(row):
    '''
    This functions specifies what makes a segment invalid
    Input: row defining an audio segment with the following columns:
        - ['meeting_id', 'part_id', 'chan', 'start', 'end', 'length', 'type', 'laugh_type']
    '''
    # If the length is shorter than min_length passed to detection algorithm, mark invalid
    #   - empirically tested -> this doesn't apply to many segments
    return (row['length'] < cfg.model["min_length"] or row['laugh_type'] == 'breath-laugh')


def append_to_index(index, row, meeting_id, part_id):
    '''
    Append this segment to invalid segments index
    '''
    start = utils.to_frames(row['start'])
    end = utils.to_frames(row['end'])

    # Append to existing intervals or create new dict entry
    if part_id in index[meeting_id].keys():
        index[meeting_id][part_id] = index[meeting_id][part_id] | P.closed(
            start, end)
    else:
        index[meeting_id][part_id] = P.closed(start, end)

    index[meeting_id]['tot_len'] += row['length']
    index[meeting_id]['tot_events'] += 1
    return index


def create_laugh_index(df):
    """
    Creates a laugh_index with all transcribed laughter events
    per particpant per meeting
    The segments are stored as disjunction of closed intervals (using portion library)
    dict structure:
    {
        meeting_id: {
            tot_len: INT,
            tot_events: INT,
            part_id: P.closed(start,end) | P.closed(start,end),
            part_id: P.closed(start,end)| P.closed(start,end)
        }
        ...
    }
    """
    global laugh_index, invalid_index

    if invalid_index == {}:
        raise RuntimeError(
            "INVALID_INDEX needs to be created before LAUGH_INDEX")
    meeting_groups = df.groupby(['meeting_id'])

    for meeting_id, meeting_df in meeting_groups:
        laugh_index[meeting_id] = {}
        laugh_index[meeting_id]['tot_len'] = 0
        laugh_index[meeting_id]['tot_events'] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values('start').groupby(['part_id'])
        for part_id, part_df in part_groups:
            laugh_index[meeting_id][part_id] = P.empty()
            for _, row in part_df.iterrows():
                # If segment is invalid, append to invalid segments index
                if seg_invalid(row):
                    invalid_index = append_to_index(
                        invalid_index, row, meeting_id, part_id)
                    continue

                # If segment is valid, append to laugh segments index
                laugh_index = append_to_index(
                    laugh_index, row, meeting_id, part_id)


def create_invalid_index(df):
    global invalid_index
    """
    Creates an invalid_index with all segments invalid for our project
    e.g. transcribed laughter events occurring next to other sounds
    The segments are stored as disjunction of closed intervals (using portion library) per participant per meeting
    dict structure (same as laugh_index):
    {
        meeting_id: {
            tot_len: INT,
            tot_events: INT,
            part_id: P.closed(start,end) | P.closed(start,end),
            part_id: P.closed(start,end) | P.closed(start,end)
        }
        ...
    }
    """
    meeting_groups = df.groupby(['meeting_id'])
    for meeting_id, meeting_df in meeting_groups:
        invalid_index[meeting_id] = {}
        invalid_index[meeting_id]['tot_len'] = 0
        invalid_index[meeting_id]['tot_events'] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values('start').groupby(['part_id'])
        for part_id, part_df in part_groups:
            for _, row in part_df.iterrows():
                invalid_index = append_to_index(
                    invalid_index, row, meeting_id, part_id)


def create_valid_speech_index():
    # TODO: Not used at the moment
    '''
    Index of those intervals that contain valid speech.

    Take whole audio files for each participant for each meeting and subtract
    all laughter and invalid segments.
    dict_structure (same as laugh_index - without tot_events)
    {
        meeting_id: {
            tot_len: INT,
            part_id: P.closed(start,end) | P.closed(start,end),
            part_id: P.closed(start,end) | P.closed(start,end)
        }
        ...
    }
    '''
    global valid_speech_index
    for _, row in parse.info_df.iterrows():
        if row.meeting_id not in valid_speech_index.keys():
            valid_speech_index[row.meeting_id] = {}

        end_frame = utils.to_frames(row.length)
        full_interval = P.closed(0, end_frame)
        valid_speech = (full_interval
                        - laugh_index[row.meeting_id].get(row.part_id, P.empty())
                        - invalid_index[row.meeting_id].get(row.part_id, P.empty()))
        valid_speech_index[row.meeting_id][row.part_id] = valid_speech
        valid_speech_index[row.meeting_id]['tot_length'] = utils.p_len(
            valid_speech)


#############################################
# EXECUTED ON IMPORT
#############################################

# First create invalid index and then laughter index
# Invalid index needs to be created first since filtered out segments
# during laugh index creation are added to the invalid index
create_invalid_index(parse.invalid_df)
create_laugh_index(parse.laugh_only_df)
