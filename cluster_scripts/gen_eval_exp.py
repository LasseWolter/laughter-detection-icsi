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

PREAMBLES_PATH = '../data/icsi/transcripts/preambles.mrt'

# Maps each meeting key to the corresponding channels used in that meeting
CHAN_AUDIO_IN_MEETING = parse_preambles(PREAMBLES_PATH)

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/icsi/data'
#base_call = (f"python main.py -i {DATA_HOME}/input -o {DATA_HOME}/output " #             "--epochs 50")
base_call = (f"python segment_laughter.py --save_to_textgrid=True --save_to_audio_files=False --config=resnet_base")

meetings = PARTITIONS['dev'] 
# lengths = [0.2]
# thresholds = [0.2,0.4,0.6,0.8]
# 
# settings = [(mt, aud, ln, thr) for mt in meetings for aud in CHAN_AUDIO_IN_MEETING[mt] 
#             for ln in lengths for thr in thresholds ]

settings = [(mt, aud) for mt in meetings for aud in CHAN_AUDIO_IN_MEETING[mt]]

output_file = open("eval_exp.txt", "w")
exp_counter = 0

#for mt, aud, ln, thr in settings:   
for mt, aud in settings:   
    exp_counter+= 1
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    #print(str(ln), str(thr), mt, aud)
    print(mt, aud)
    expt_call = (
        f"{base_call} "
        f"--input_audio_file={DATA_HOME}/speech/dev/{mt}/{aud} "
        f"--output_dir={DATA_HOME}/eval_output/{mt} "
        #f"--min_length={ln} "
        #f"--threshold={thr} "
        f"--model_path=checkpoints/icsi_eval"
    )
    print(expt_call, file=output_file)

output_file.close()

print(f'Generated {exp_counter} experiments')
print(f' - {len(meetings)} meetings')
print(f'    - each with a number of audio channels')
#print(f' - {len(thresholds)} thresholds')
#print(f' - {len(lengths)} min_lengths')
