"""
This is basically the Vad-Dataset from lhotse (lhotse/dataset/vad.py) with minor adjustments
It changes the is_voice field to an is_laugh field which states if the corresponding cut is a laughter cut or not
Note - for now the laughter cuts ONLY contain laughter and the speech cuts ONLY contain speech
  -> there are NO segments that contains both (might be worth trying for better detection of boundaries)

The is_laugh tensor is created by traversing the cuts in the CutSet and taking the value in supervisions[0].custom['is_laugh']
  -> supervisions[0] because there is only one supervision per segment (as they are manually created by load_data.py)
"""

from typing import Callable, Dict, Sequence

import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone
import torch
import numpy as np
import config as cfg


class LadDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the laugh activity detection task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': (B x T x F) tensor
            'input_lens': (B,) tensor
            'is_laugh': (T x 1) tensor
            'cut': List[Cut]
        }
    """

    def __init__(
        self,
        input_strategy: BatchIO = PrecomputedFeatures(),
        cut_transforms: Sequence[Callable[[CutSet], CutSet]] = None,
        input_transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_strategy = input_strategy
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        # Not validating because the meta data of truncated cuts won't match
        # the feature/recording data until  load_features() / load_audio() is called
        # validate(cuts)
        # cuts = cuts.sort_by_duration()
        for tfnm in self.cut_transforms:
            cuts = tfnm(cuts)
        inputs, input_lens = self.input_strategy(cuts)
        for tfnm in self.input_transforms:
            inputs = tfnm(inputs)
        is_laugh = []
        for c in cuts:
            # Assumes that there is only one supervision per cut
            is_laugh.append(c.supervisions[0].custom["is_laugh"])
        return {
            "inputs": inputs,
            "input_lens": input_lens,
            "is_laugh": torch.tensor(is_laugh, dtype=torch.int32),
            "cut": cuts,
        }


class InferenceDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the inference of laughter detection.
    """

    def __init__(self, feats, n_frames=cfg.FEAT["num_samples"]) -> None:
        super().__init__()
        self.feats = feats  # The feature representation for the whole meeting
        self.n_frames = n_frames

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        ret = self.feats[index : index + self.n_frames]
        # If we get to the end of the audio sample we need to pad to keep
        # the sample number at 100
        if ret.shape[0] != cfg.FEAT["num_samples"]:
            pad_amount = cfg.FEAT["num_samples"] - ret.shape[0]
            # zero pad first axis only on the right
            ret = np.pad(ret, ((0, pad_amount), (0, 0)))
        return ret
