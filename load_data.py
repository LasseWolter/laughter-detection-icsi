from torch.utils.data import DataLoader
from lhotse import CutSet, Fbank, FbankConfig, Recording, MonoCut
from lhotse.dataset import SingleCutSampler
from lhotse import RecordingSet
from lad import LadDataset, InferenceDataset
import config as cfg
import os
import sys
sys.path.append('./utils/')
from utils import get_feat_extractor

def create_training_dataloader(cutset_dir, split, shuffle=False):
    '''
    Create a dataloader for the provided split 
        - split needs to be one of 'train', 'dev' and 'test'
        - cutset location is the directory in which the lhotse-CutSet with all the information about cuts and their features is stored
        - shuffle allows shuffling the cutset before the dataloader is created from it
    '''
    if split not in ['train', 'dev', 'test']:
        raise ValueError(
            f"Unexpected value for split. Needs to be one of 'train, dev, test'. Found {split}")

    # Load cutset for split
    cuts = CutSet.from_jsonl(os.path.join(
        cutset_dir, f'{split}_cutset_with_feats.jsonl'))
    
    if shuffle:
        cuts = cuts.shuffle()

    # Construct a Pytorch Dataset class for Laugh Activity Detection task:
    dataset = LadDataset()
    sampler = SingleCutSampler(cuts, max_cuts=32)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
    return dataloader


def create_inference_dataloader(audio_path):
    '''
    Create inference dataloader for audio_file passed using `audio_path`. 
    Loads the audio as one single cut and creates a feature representation of it.
    These features are then used to created an Inference Dataset from which the 
    features for small windows can be read one by one
    '''
    rec = Recording.from_file(audio_path)
    cut = MonoCut(id='inference-cut', start=0.0, duration=rec.duration, channel=0, recording=rec)

    extractor = get_feat_extractor(num_samples=cfg.FEAT['num_samples'], num_filters=cfg.FEAT['num_filters'])
    # f2 = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))
    feats_all = cut.compute_features(extractor)

    # Construct a Pytorch Dataset class for inference using the
    dataset = InferenceDataset(feats_all)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader