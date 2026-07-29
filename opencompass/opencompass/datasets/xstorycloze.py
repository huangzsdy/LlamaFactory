"""XStoryCloze - Multilingual Story Cloze dataset"""
from datasets import load_dataset
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class XStoryClozeDataset(BaseDataset):
    """XStoryCloze (multi-language story completion)"""
    LANGUAGES = ['en', 'zh', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'pl', 'ko']
    
    @staticmethod
    def load(path, lang='en'):
        return load_dataset(path, lang)