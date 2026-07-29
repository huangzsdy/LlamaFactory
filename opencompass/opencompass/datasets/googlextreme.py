"""Google XTREME dataset"""
from datasets import load_dataset
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class GoogleXTREMEDataset(BaseDataset):
    """Google XTREME multilingual benchmark"""
    SUBSETS = ['xquad', 'tydiqa', 'marc', 'xnli', 'pawsx', 'bucc', 'paren']
    
    @staticmethod
    def load(path, subset='xnli'):
        return load_dataset(path, subset)