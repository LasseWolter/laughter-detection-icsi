from lhotse import Fbank, FbankConfig, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.features.kaldifeat import KaldifeatMelOptions, KaldifeatFrameOptions
import torch


def get_feat_extractor(num_samples, num_filters, use_kaldi=False):
    """
    Returns an feature extractor with passed config
        - if possible returns a GPU compatible extractor
        - o/w returns the normal Fbank-extractor which runs on CPU

    use_kaldi=True will always try returning the Kaldifeat-Extractor
    """
    # Frame shift is given in seconds. Thus, dived 1s by the number of samples specified in config
    frame_shift = 1 / num_samples
    if use_kaldi or torch.cuda.is_available():
        try:
            print("Trying to use Kaldifeat-Extractor...")
            extractor = KaldifeatFbank(
                KaldifeatFbankConfig(
                    device="cuda",
                    frame_opts=KaldifeatFrameOptions(frame_shift=frame_shift),
                    mel_opts=KaldifeatMelOptions(num_bins=num_filters),
                )
            )
        except:
            print("Couldn't use Kaldifeat-Extractor. Using CPU-FbankExtractor")

    extractor = Fbank(FbankConfig(num_filters=num_filters, frame_shift=frame_shift))
    return extractor
