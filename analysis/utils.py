import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ANALYSIS as cfg
import portion as P


def to_frames(time_in_sec):
    '''
    Represent time in seconds as number of frames.
    Frame duration is defined in config
    '''
    # Calculate fps (1000ms/frame_duration)
    factor = 1000/cfg['model']['frame_duration']
    return round(time_in_sec*factor)


def to_sec(num_of_frames):
    '''
    Turn time in number of frames to time in seconds.
    Frame duration is defined in config
    '''
    # Calculate fps (1000ms/frame_duration)
    factor = 1000/cfg['model']['frame_duration']
    return num_of_frames/factor


def p_len(p_interval):
    '''
    Takes an interval of portion's Interval class and returns its (accumulated) length.
    Portion's Interval class includes disjunctions of atomic intervals.

    E.g. p_len( (P.openclosed(1,3) | P.openclosed(10,11)) ) = 5
    '''
    # Iterate over the (disjunction of) interval(s) with step-size 1
    # Then count the number of elements in the list
    return len(list(P.iterate(p_interval, step=1)))


def seg_overlaps(seg, indices, meeting_id, part_id):
    '''
    Returns true if passed segment overlaps with any segment of this participant
    in any of the passed indices 
    Input: segment, list of indices, meeting_id, part_id
    '''
    for index in indices:
        # If no entry exists for this meeting_id or this part_id return False
        if (meeting_id not in index.keys() or part_id not in index[meeting_id].keys()):
            continue
        if seg.overlaps(index[meeting_id].get(part_id, P.empty())):
            return True

    return False
