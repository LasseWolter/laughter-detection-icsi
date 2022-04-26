#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
from lxml import etree
from dotenv import load_dotenv, find_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ANALYSIS as cfg

load_dotenv(find_dotenv(".env"))
FEATS_DIR = os.getenv("FEATS_DIR")

# Name of the experiment
NAME = os.getenv("NAME") if os.getenv("NAME") != None else "exp1"
NUM_EPOCHS = int(os.getenv("EPOCHS"))


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


PREAMBLES_PATH = os.path.join(cfg["transcript_dir"], "preambles.mrt")
CHAN_AUDIO_IN_MEETING = parse_preambles(PREAMBLES_PATH)

# The home dir on the node's scratch disk
USER = os.getenv("USER")
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

DATA_HOME = f"{SCRATCH_HOME}/icsi/data"
# base_call = (f"python main.py -i {DATA_HOME}/input -o {DATA_HOME}/output " #             "--epochs 50")
base_call = f"python train.py --config resnet_base --checkpoint_dir {SCRATCH_HOME}/icsi/checkpoints --data_root {SCRATCH_HOME}/icsi --lhotse_dir {FEATS_DIR} --num_epochs=1"

out_file_name = f"{NAME}_exp_train.txt"
output_file = open(out_file_name, "w")
exp_counter = 0

for i in range(NUM_EPOCHS):
    exp_counter += 1
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = f"{base_call} "
    print(expt_call, file=output_file)

output_file.close()

print(
    f"Generated commands for {exp_counter} training epochs.\nSaved in: {out_file_name}"
)
