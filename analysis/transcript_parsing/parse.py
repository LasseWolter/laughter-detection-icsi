from typing import List, Optional, Tuple
from pydantic import BaseModel
from strenum import StrEnum
from lxml import etree

# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import os
import pandas as pd

chan_to_part = {}  # index mapping channel to participant per meeting
part_to_chan = {}  # index mapping participant to channel per meeting
laugh_only_df = pd.DataFrame()  # dataframe containing transcribed laugh only events
invalid_df = pd.DataFrame()  # dataframe containing invalid segments

# Dataframe containing total length and audio_path of each channel
info_df = pd.DataFrame()


class SegmentType(StrEnum):
    '''
    Describes the type of data that was transcribed in this segment. 
    For detailed information: https://www1.icsi.berkeley.edu/Speech/mr/icsimc_doc/trans_guide.txt
    '''
    INVALID = 'invalid'  # e.g. laughter segments occurring next to speech or other noise
    SPEECH = 'speech' 
    LAUGH = 'laugh' 
    OTHER_VOCAL = 'other_vocal'  # segments containing a single VocalSound that's not laughter
    NON_VOCAL = 'non_vocal' # segments containing a single NonVocalSound (e.g. 'mic noise')
    MIXED = 'mixed'  # contains some mixture of speech / noise and silence (but no laughter)


class Segment(BaseModel):
    """Represent a Transcription segment from ICSI transcripts"""

    meeting_id: str
    part_id: str
    chan_id: str
    start: float
    end: float
    length: float
    type: SegmentType
    laugh_type: Optional[str]


def parse_preambles(path):
    global chan_to_part, part_to_chan
    """
    Creates 2 id mappings
    1) Dict: (meeting_id) -> (dict(chan_id -> participant_id))
    2) Dict: (meeting_id) -> (dict(participant_id -> chan_id))
    """
    dirname = os.path.dirname(__file__)
    preambles_path = os.path.join(dirname, path)
    chan_to_part = {}
    tree = etree.parse(preambles_path)
    meetings = tree.xpath("//Meeting")
    for meeting in meetings:
        id = meeting.get("Session")
        part_map = {}
        # Make sure that both Name and Channel attributes exist
        for part in meeting.xpath(
            "./Preamble/Participants/Participant[@Name and @Channel]"
        ):
            part_map[part.get("Channel")] = part.get("Name")

        chan_to_part[id] = part_map

    part_to_chan = {}
    for meeting_id in chan_to_part.keys():
        part_to_chan[meeting_id] = {
            part_id: chan_id for (chan_id, part_id) in chan_to_part[meeting_id].items()
        }


def xml_to_segment(xml_seg, meeting_id: str) -> Optional[Segment]:
    """
    Input: xml laughter segment as etree Element, meeting id
    Output: list of features representing this laughter segment:
        - Format: [meeting_id, part_id, chan_id, start, end, length, l_type]
        - returns [] if no valid part_id was found
    """
    part_id = xml_seg.get("Participant")
    # If participant doesn't have a corresponding audio channel
    # discard this segment
    if part_id not in part_to_chan[meeting_id].keys():
        return None

    chan_id = part_to_chan[meeting_id][part_id]
    start = float(xml_seg.get("StartTime"))
    end = float(xml_seg.get("EndTime"))
    length = end - start

    seg_type, laugh_type = _get_segment_type(xml_seg)

    new_seg = Segment(
        meeting_id=meeting_id,
        part_id=part_id,
        chan_id=chan_id,
        start=start,
        end=end,
        length=length,
        type=seg_type,
        laugh_type=laugh_type,
    )
    return new_seg


def _get_segment_type(xml_seg) -> Tuple[SegmentType, str]:
    """
    Determine the segment type of passed xml-segment - if laughter also return type of laughter
    Input: xml-segment of the shape of ICSI transcript segments
    Output: [segment_type, laugh_type]
    """
    children = xml_seg.getchildren()
    laugh_type  = None
    seg_type = SegmentType.MIXED

    if len(children) == 0:
        seg_type = SegmentType.SPEECH
    elif len(children) == 1:
        child = children[0]
        if child.tag == "VocalSound":
            if "laugh" in child.get("Description"):
                # Check that there is no text in any sub-element of this tag
                # which meant speech occurring next to laughter
                if "".join(xml_seg.itertext()).strip() == "":
                    seg_type = SegmentType.LAUGH
                    laugh_type = child.get("Description")
                else:
                    seg_type = SegmentType.INVALID

            else:
                seg_type = SegmentType.OTHER_VOCAL

        elif child.tag == "NonVocalSound":
            seg_type = SegmentType.NON_VOCAL

        else:
            # This is because there are also tags like <Comment>
            seg_type = SegmentType.SPEECH
    else:
        # Track laughter next to speech or noise to discard these segments from evaluation
        # If laughter occurs next to speech we can properly track it but it's still laughter
        # Thus a prediction on such a segment shouldn't be considered wrong but just be ignored.
        laughs = xml_seg.xpath("./VocalSound[contains(@Description, 'laugh')]")
        
        # If one of VocalSound or NonVocalSound tags appear, classify as mixed
        tag_types = list(map(lambda x: x.tag, children))
        if laughs != []:
            seg_type = SegmentType.INVALID
        elif "NonVocalSound" in tag_types or "VocalSound" in tag_types: 
            seg_type = SegmentType.MIXED
        else:
            seg_type = SegmentType.SPEECH

    return (seg_type, laugh_type)


def get_segment_list(filename, meeting_id):
    """
    Returns four lists of Segment objects represented as dict:
        1) List containing segments laughter only (no text or other sounds surrounding it)
        2) List containing invalid segments (e.g. laughter surrounding by other sounds)
        3) List containing speech segments
        4) List containing noise segments
    """
    # Comment shows which types of segments each list will hold
    invalid_list: List[Segment] = []  # INVALID
    laugh_only_list: List[Segment] = []  # LAUGH
    speech_list: List[Segment] = []  # SPEECH
    noise_list: List[Segment] = []  # MIXED, NON_VOCAL, OTHER_VOCAL

    # Get all segments that contain some kind of laughter (e.g. 'laugh', 'breath-laugh')
    # xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')]]"
    tree = etree.parse(filename)
    # laugh_segs = tree.xpath(xpath_exp)
    all_segs = tree.xpath("//Segment")

    # For each laughter segment classify it as laugh only or mixed laugh
    # mixed laugh means that the laugh occurred next to speech or any other sound
    for xml_seg in all_segs:
        seg = xml_to_segment(xml_seg, meeting_id)
        if (seg==None): # Skip segment without audio chan
            continue
        if seg.type == SegmentType.LAUGH:
            laugh_only_list.append(seg.dict())
        elif seg.type == SegmentType.SPEECH:
            speech_list.append(seg.dict())
        elif seg.type == SegmentType.INVALID:
            invalid_list.append(seg.dict())
        else:
            noise_list.append(seg.dict())

    return invalid_list, speech_list, laugh_only_list, noise_list


def general_info_to_list(filename, meeting_id):
    general_info_list = []
    tree = etree.parse(filename)
    # Get End-Timestamp of last transcription of the meeting
    meeting_len = tree.findall("//Segment")[-1].get("EndTime")
    for chan_id, part_id in chan_to_part[meeting_id].items():
        path = os.path.join(meeting_id, f"{chan_id}.sph")
        general_info_list.append([meeting_id, part_id, chan_id, meeting_len, path])

    return general_info_list


def get_transcripts(path):
    """
    Parse meeting transcripts and store laughs in laugh_df
    """

    files = []
    # If a directory is given take all .mrt files
    # otherwise only take given file
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)

    # Match particular file or all .mrt files
    if os.path.isdir(path):
        for filename in os.listdir(path):
            # All icsi meetings have a 6 letter ID (-> split strips the .mrt extension)
            if filename.endswith(".mrt") and len(filename.split(".")[0]) == 6:
                files.append(filename)
    else:
        if path.endswith(".mrt"):
            files.append(path)

    return files


def create_dfs(file_dir, files):
    """
    Creates four segment-dataframes and one info-dataframe:
        1) laugh_only_df: dataframe containing laughter only snippets
        2) invalid_df: containing snippets with laughter next to speech (discarded from evaluation)
        3) speech_df: contains speech
        4) noise_df: containing all other segments
            - NOTE: this doesn't include silence as silence happens mostly BETWEEN transcription segments

        5) general_info_df: dataframe containing total length and audio_path of each channel

    segment_dataframes columns: ['meeting_id', 'part_id', 'chan_id', 'start', 'end', 'length', 'type', 'laugh_type']

    info_df columns: ['meeting_id', 'part_id', 'chan_id', 'length', 'path']

    """
    global laugh_only_df, invalid_df, info_df, noise_df, speech_df

    # Define lists holding all the rows for those dataframes
    tot_invalid_segs = []
    tot_speech_segs= []
    tot_laugh_only_segs = []
    tot_noise_segs = []

    general_info_list = []
    # Iterate over all .mrt files
    for filename in files:
        # Get meeting id by getting the basename and stripping the extension
        basename = os.path.basename(filename)
        meeting_id = os.path.splitext(basename)[0]
        full_path = os.path.join(file_dir, filename)

        general_info_sublist = general_info_to_list(full_path, meeting_id)
        general_info_list += general_info_sublist

        invalid, speech, laugh_only, noise = get_segment_list(full_path, meeting_id)
        tot_invalid_segs += invalid
        tot_speech_segs += speech 
        tot_laugh_only_segs += laugh_only
        tot_noise_segs += noise

    laugh_only_df = pd.DataFrame(tot_laugh_only_segs)
    invalid_df = pd.DataFrame(tot_invalid_segs )
    speech_df = pd.DataFrame(tot_speech_segs)
    noise_df = pd.DataFrame(tot_noise_segs)

    # Create info_df with specified columns and dtypes
    info_dtypes = {"length": "float"}
    info_cols = ["meeting_id", "part_id", "chan_id", "length", "path"]
    info_df = pd.DataFrame(general_info_list, columns=info_cols)
    info_df = info_df.astype(dtype=info_dtypes)


def parse_transcripts(path):
    """
    Function executed on import of this module.
    Parses transcripts (including preamble.mrt) and creates:
        - chan_to_part: index mapping channel to participant per meeting
        - part_to_chan: index mapping participant to channel per meeting
        - laugh_only_df: dataframe containing transcribed laugh only events
        - invalid_df: dataframe containing invalid segments (e.g. mixed laugh and speech)
    """
    parse_preambles(os.path.join(path, "preambles.mrt"))

    transc_files = get_transcripts(path)
    create_dfs(path, transc_files)


def _print_stats(df):
    """
    Print stats of laugh_df - for information/debugging only
    """
    print(df)
    if df.size == 0:
        print("Empty DataFrame")
        return
    print("avg-snippet-length: {:.2f}s".format(df["length"].mean()))
    print("Number of snippets: {}".format(df.shape[0]))
    tot_dur = df["length"].sum()
    print(
        "Accumulated segment duration in three formats: \n- {:.2f}h \n- {:.2f}min \n- {:.2f}s".format(
            (tot_dur / 3600), (tot_dur / 60), tot_dur
        )
    )


def main():
    """
    Main executed when file is called directly
    NOT on import
    """
    file_path = os.path.join(os.path.dirname(__file__), "data")
    parse_transcripts(file_path)

    print("\n----INALID SEGMENTS-----")
    _print_stats(invalid_df)

    print("\n----SPEECH SEGMENTS-----")
    _print_stats(speech_df)
    
    print("\n----LAUGHTER ONLY-----")
    _print_stats(laugh_only_df)

    print("\n----NOISE SEGMENTS-----")
    _print_stats(noise_df)

    print("\n----INFO DF-----")
    print(info_df)


if __name__ == "__main__":
    main()


#############################################
# EXECUTED ON IMPORT
#############################################
# Parse transcripts on import
file_path = os.path.join(os.path.dirname(__file__), "data")
parse_transcripts(file_path)
