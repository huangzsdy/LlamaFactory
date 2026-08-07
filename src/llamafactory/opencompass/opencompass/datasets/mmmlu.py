# flake8: noqa
# yapf: disable

import json
import os.path as osp
import os
from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

import pandas as pd
from pathlib import Path

@LOAD_DATASET.register_module()
class MMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        subset = name.split('_')[1].replace('-', '_')
        print(f"path:{path}, subset:{subset}, Path(path):{Path(path)}")
        for split in ['test']:
            csv_path = os.path.join(path, 'test', f'{name}.csv')
            df = pd.read_csv(csv_path)
            data_list = []


            for index, item in df.iterrows():
                # print(f"item:{item} !!!!")
                data_list.append({
                    'input': item['Question'],
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                    'target': item['Answer'],
                    'subject': item['Subject'].replace('_', ' ')
                })
            dataset[split] = Dataset.from_list(data_list)
        return dataset
    # @staticmethod
    # def load(path: str, name: str):
    #     dataset = DatasetDict()
    #     subset = name.split('_')[1].replace('-', '_')
    #     print(f"path:{path}, subset:{subset}, Path(path):{Path(path)}")
    #     # for split in ['test']:
    #     for file in Path(path).glob(f'*.csv'):
    #         # data = load_dataset(path=path,
    #         #                     name=subset,
    #         #                     split=split,
    #         #                     trust_remote_code=True)
    #         data = pd.read_csv(file)
    #         dataset_list = []
    #         # for item in data:
    #         for index, item in data.iterrows():
    #             # print(f"item:{item} !!!!")
    #             dataset_list.append({
    #                 'input': item['Question'],
    #                 'A': item['A'],
    #                 'B': item['B'],
    #                 'C': item['C'],
    #                 'D': item['D'],
    #                 'target': item['Answer'],
    #                 'subject': item['Subject'].replace('_', ' ')
    #             })
    #         # dataset[split] = Dataset.from_list(dataset_list)
    #         dataset['test'] = Dataset.from_list(dataset_list)
    #     return dataset

@LOAD_DATASET.register_module()
class MMMLULiteDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=False)
        dataset = DatasetDict()
        name = name.split('_')[-1]
        raw_data = []
        filename = osp.join(path, name, 'test.jsonl')
        with open(filename, encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset
