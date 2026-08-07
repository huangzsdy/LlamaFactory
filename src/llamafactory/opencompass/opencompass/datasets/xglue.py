"""XGLUE dataset - A Benchmark for Multilingual Language Understanding"""
from datasets import load_dataset
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class XGLUEDataset(BaseDataset):
    """XGLUE: Xtreme multilingual multi-task benchmark"""
    SUBSETS = ['xcopa', 'XNLI', 'ner', 'pos', 'q2q', 'qam', 'qe', 'mlqa', 'tydiqa', 'pawsx']
    
    @staticmethod
    def load(path, subset='xcopa'):
        return load_dataset(path, subset)