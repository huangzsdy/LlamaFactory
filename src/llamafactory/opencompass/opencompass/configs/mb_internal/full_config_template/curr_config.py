import os.path as osp
from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from ..dataset_collections.custom_mmmlu_mgsm import  hf_infer_datasets_total, datasets_map
    #from ..summarizers.core_all_v3 import summarizer
    from ..summarizers.custom_mmmlu_mgsm_summarizer import summarizer
    from ..models.small_models.minicpm_base_model import base_vllm_model, base_hf_model
    from ..cluster.local_cluster import eval, infer_num_worker as infer

models_path = [
    "/data/multilingual_projects/training_data/ja_wiki/20260505_202015/ja_transferred_ckpt_2000/minicpm-1b/4k/demo-iter_2000",
]

env_languages="ja"
languages=env_languages.split(',')
print(f"----------------------------------------------------languages:{languages}")

language_dataset_map = datasets_map


# NOTE: num_gpus_per_infer should equal to the GPU nums
# when the task is begin, cannot change it!
num_gpus_per_infer = 1
num_gpus_per_model = 1

infer['runner']['max_num_workers'] = num_gpus_per_infer
infer['runner']['max_workers_per_gpu'] = num_gpus_per_model


infer['partitioner']['num_worker'] = num_gpus_per_infer // num_gpus_per_model


vllm_models = []
hf_models = []

for model_path in models_path:
    if model_path.endswith('/'):
        model_path = model_path[:-1]
    # assume dir name is hqdata_exp_{JOB_ID}.decay.{ITER}
    dir_name = osp.basename(model_path)
    abbr = dir_name.split('-')[0]
    ckpt_iter = dir_name.split('-')[-1]

    summarizer_abbr = f'{abbr}@{ckpt_iter}'

    tmp_hf_model_cfg = deepcopy(base_hf_model)
    tmp_hf_model_cfg['abbr'] = f'{abbr}@{ckpt_iter}-hf'
    tmp_hf_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_hf_model_cfg['path'] = model_path
    hf_models.append(tmp_hf_model_cfg)

    tmp_vllm_model_cfg = deepcopy(base_vllm_model)
    tmp_vllm_model_cfg['abbr'] = f'{abbr}@{ckpt_iter}-vllm'
    tmp_vllm_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_vllm_model_cfg['path'] = model_path
    vllm_models.append(tmp_vllm_model_cfg)

model_dataset_combinations = []
models = []
datasets = []

target_datasets= []
if languages == ['all']:
    target_datasets = [ x for sub in language_dataset_map.values() for x in sub ]
    print(f"target_datasets:{target_datasets}")
    print(f"len(target_datasets):{len(target_datasets)}")
    print(f"target_datasets[0]:{target_datasets[0]}")
else:
    target_datasets = []
    for lang in languages:
        if lang not in language_dataset_map.keys():
            print(f"{lang} not in :{language_dataset_map.keys()}, continue !!!!")
            continue
        target_datasets.extend(language_dataset_map[lang])
        print(f"****************lang:{lang}, target_datasets:{target_datasets}")
    # contains all languages if not match any language
    if not target_datasets:
        target_datasets = language_dataset_map.values()


model_dataset_combinations.append(dict(models=vllm_models, datasets=target_datasets))
models.extend(vllm_models)
datasets.extend(target_datasets)


#model_dataset_combinations.append(dict(models=vllm_models, datasets=base_datasets_total))
#models.extend(vllm_models)
#datasets.extend(base_datasets_total)

model_dataset_combinations.append(dict(models=hf_models, datasets=hf_infer_datasets_total))
models.extend(hf_models)
datasets.extend(hf_infer_datasets_total)
