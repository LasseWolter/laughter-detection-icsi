#!/usr/bin/env python3
"""
Script for generating .txt file containing experiments for evaluation for each track of the devlopment set.
The location of the ML-model used for evaluation is set in the MODEL_PATH global variable .

Generate one experiment per line, e.g.:
`python segment_laughter.py --save_to_textgrid=True --save_to_audio_files=False --config=resnet_base --input_audio_file=/disk/scratch/s1660656/icsi/data/speech/dev/Bmr021/chan0.sph --output_dir=/disk/scratch/s1660656/icsi/data/eval_output/Bmr021 --model_path=checkpoints/icsi_eval`
"""
import os
from lxml import etree
import numpy as np

# Define split for evaluation (one of 'train', 'dev' and 'test')
SPLIT = "dev"

# Copy over all data if train split is used
# (didn't create an additional train dir to avoid lots of duplicate data)
# Otherwise only copy dev or test data
if SPLIT == "train":
    SPLIT_DIR = "all"
else:
    SPLIT_DIR = SPLIT

# Path of the model's checkpoint
MODEL_PATH = "checkpoints/icsi_eval"

# The model config needs to be the name of one of the configs in configs.py
MODEL_CONFIG = "resnet_base"

# Settings as numeric list
lower_range = np.linspace(0, 0.9, 19).round(2)
upper_range = np.linspace(0.91, 1, 10).round(2)
thrs = np.concatenate((lower_range, upper_range))
min_lens = [0, 0.1, 0.2]
# Comma-separated list of settings to try
THRESHOLDS = ",".join([str(t) for t in thrs])
MIN_LENGTHS = ",".join([str(l) for l in min_lens])

OUTPUT_FILE = f"eval_{SPLIT}_exp.txt"


def parse_preambles(filename):
    """
    Input: filepath of the preambles.mrt
    Output: Dict mapping meeting_ids to list of audio_files (of individual  channels) present in this meeting
        1) Dict: (meeting_id) -> [chan_id_1_1, chan_id_1_2,...]
        2) Dict: (meeting_id) -> [chan_id_2_1, chan_id_2_2,...]
    """
    chan_audio_in_meeting = {}
    tree = etree.parse(filename)
    meetings = tree.xpath("//Meeting")
    for meeting in meetings:
        id = meeting.get("Session")
        for part in meeting.xpath("./Preamble/Channels/Channel"):
            if id in chan_audio_in_meeting.keys():
                chan_audio_in_meeting[id].append(part.get("AudioFile"))
            else:
                chan_audio_in_meeting[id] = [part.get("AudioFile")]

    return chan_audio_in_meeting


# Taken from lhotse icsi recipe to minimise speaker overlap
PARTITIONS = {
    "train": [
        "Bdb001",
        "Bed002",
        "Bed003",
        "Bed004",
        "Bed005",
        "Bed006",
        "Bed008",
        "Bed009",
        "Bed010",
        "Bed011",
        "Bed012",
        "Bed013",
        "Bed014",
        "Bed015",
        "Bed016",
        "Bed017",
        "Bmr001",
        "Bmr002",
        "Bmr003",
        "Bmr005",
        "Bmr006",
        "Bmr007",
        "Bmr008",
        "Bmr009",
        "Bmr010",
        "Bmr011",
        "Bmr012",
        "Bmr014",
        "Bmr015",
        "Bmr016",
        "Bmr019",
        "Bmr020",
        "Bmr022",
        "Bmr023",
        "Bmr024",
        "Bmr025",
        "Bmr026",
        "Bmr027",
        "Bmr028",
        "Bmr029",
        "Bmr030",
        "Bmr031",
        "Bns002",
        "Bns003",
        "Bro003",
        "Bro004",
        "Bro005",
        "Bro007",
        "Bro008",
        "Bro010",
        "Bro011",
        "Bro012",
        "Bro013",
        "Bro014",
        "Bro015",
        "Bro016",
        "Bro017",
        "Bro018",
        "Bro019",
        "Bro022",
        "Bro023",
        "Bro024",
        "Bro025",
        "Bro026",
        "Bro027",
        "Bro028",
        "Bsr001",
        "Btr001",
        "Btr002",
        "Buw001",
    ],
    "dev": ["Bmr021", "Bns001"],
    "test": ["Bmr013", "Bmr018", "Bro021"],
}

PREAMBLES_PATH = "../data/icsi/transcripts/preambles.mrt"

# Maps each meeting key to the corresponding channels used in that meeting
CHAN_AUDIO_IN_MEETING = parse_preambles(PREAMBLES_PATH)

# The home dir on the node's scratch disk
USER = os.getenv("USER")

# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

DATA_HOME = f"{SCRATCH_HOME}/icsi/data"

base_call = (
    f"python segment_laughter.py --save_to_textgrid=True --save_to_audio_files=False"
    f" --config={MODEL_CONFIG} --model_path={MODEL_PATH} --thresholds={THRESHOLDS} --min_lengths={MIN_LENGTHS}"
)

meetings = PARTITIONS[SPLIT]

# Parameter settings are set directly in the segment_laughter.file
#    - threshholds and min_lengths

audio_tracks = [(mt, chan) for mt in meetings for chan in CHAN_AUDIO_IN_MEETING[mt]]

output_file = open(OUTPUT_FILE, "w")
exp_counter = 0

for meeting, chan in audio_tracks:
    exp_counter += 1
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    print(meeting, chan)
    expt_call = (
        f"{base_call} "
        f"--input_audio_file={DATA_HOME}/speech/{SPLIT_DIR}/{meeting}/{chan} "
        f"--output_dir={DATA_HOME}/eval_output/{meeting} "
    )
    print(expt_call, file=output_file)

output_file.close()

print(f"Generated {exp_counter} experiments for split: {SPLIT}")
print(f" - {len(meetings)} meetings")
print(f"    - each with a number of audio channels")
