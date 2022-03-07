from pathlib import Path

model = {
    # Min-length used for parsing the transcripts
    "min_length": 0.2,

    # Frame duration used for parsing the transcripts
    "frame_duration": 10  # in ms
}

train = {
    # How long each sample for training should be 
    "subsample_duration": 1.0,  # in s
    "random_seed": 23,

    # Used in creation of train, val and test df in 'create_data_df'
    "float_decimals": 2,  # number of decimals to round floats to
    # Test uses the remaining fraction
    "train_val_test_split": [0.8, 0.1]
}

root_path = Path(__file__).absolute().parent.parent
general = {
    "transcript_dir": str(root_path / 'data/icsi/transcripts'),
    "speech_dir": str(root_path / 'data/icsi/speech')
}