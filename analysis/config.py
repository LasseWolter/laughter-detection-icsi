model = {
    "min_length": 0.2,
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
