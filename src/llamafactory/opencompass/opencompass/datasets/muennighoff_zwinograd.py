"""Muennighoff Winograd Schema Challenge dataset"""
from datasets import load_dataset
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class MuennighoffWinogradDataset(BaseDataset):
    """Muennighoff Winograd Schema Challenge (multi-language)"""
    LANGUAGES = ['en', 'zh', 'ja', 'de', 'fr', 'ru', 'es', 'it', 'pt', 'ar']
    
    @staticmethod
    def load(path, lang='en'):
        return load_dataset(path, lang)