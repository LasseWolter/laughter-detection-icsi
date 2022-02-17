#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
from lxml import etree


def parse_preambles(filename):
    '''
    Input: filepath of the preambles.mrt
    Output: Dict mapping meeting_ids to list of audio_files (of individual  channels) present in this meeting 
        1) Dict: (meeting_id) -> [chan_id_1_1, chan_id_1_2,...] 
        2) Dict: (meeting_id) -> [chan_id_2_1, chan_id_2_2,...] 
    '''
    chan_audio_in_meeting = {}
    tree = etree.parse(filename)
    meetings = tree.xpath('//Meeting')
    for meeting in meetings:
        id = meeting.get('Session')
        for part in meeting.xpath('./Preamble/Channels/Channel'):
            if id in chan_audio_in_meeting.keys():
                chan_audio_in_meeting[id].append(part.get('AudioFile')) 
            else:
                chan_audio_in_meeting[id] = [part.get('AudioFile')]

    return chan_audio_in_meeting


PREAMBLES_PATH = './data/icsi/preambles.mrt'
CHAN_AUDIO_IN_MEETING = parse_preambles(PREAMBLES_PATH)

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/icsi/data'
#base_call = (f"python main.py -i {DATA_HOME}/input -o {DATA_HOME}/output " #             "--epochs 50")
base_call = (f"python train.py --config resnet_base --checkpoint_dir {SCRATCH_HOME}/icsi/checkpoints")

lengths = [0.2]
thresholds = [0.2,0.4,0.6,0.8]

settings = [(mt, aud, ln, thr) for mt in meetings for aud in CHAN_AUDIO_IN_MEETING[mt] 
            for ln in lengths for thr in thresholds ]


output_file = open("experiment.txt", "w")
exp_counter = 0

for mt, aud, ln, thr in settings:   
    exp_counter+= 1
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    print(str(ln), str(thr), mt, aud)
    expt_call = (
        f"{base_call} "
    )
    print(expt_call, file=output_file)

output_file.close()

print(f'Generated {exp_counter} experiments')
print(f' - {len(meetings)} meetings')
print(f'    - each with a number of audio channels')
print(f' - {len(thresholds)} thresholds')
print(f' - {len(lengths)} min_lengths')
