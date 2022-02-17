from torch.utils.data import DataLoader
from lhotse import CutSet, Fbank, FbankConfig, Recording
from lhotse.dataset import SingleCutSampler
from lhotse import RecordingSet
from lad import LadDataset, InferenceDataset
import os

def create_training_dataloader(cutset_dir, split):
    '''
    Create a dataloader for the provided split 
        - split needs to be one of 'train', 'dev' and 'test'
        - cutset location is the directory in which the lhotse-CutSet with all the information about cuts and their features is stored
    '''
    if split not in ['train', 'dev', 'test']:
        raise ValueError(
            f"Unexpected value for split. Needs to be one of 'train, dev, test'. Found {split}")

    # Load cutset for split
    cuts = CutSet.from_jsonl(os.path.join(
        cutset_dir, f'{split}_cutset_with_feats.jsonl'))

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
    single_rec = Recording.from_file(audio_path)
    # TODO: Is there a better way then creating a RecordingSet and CutSet with len=1
    cuts = CutSet.from_manifests(RecordingSet.from_recordings([single_rec]))
    # Cut that contains the whole audiofile
    cut_all = cuts[0]

    f2 = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))
    feats_all = cut_all.compute_features(f2)

    # Construct a Pytorch Dataset class for inference using the
    dataset = InferenceDataset(feats_all)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader