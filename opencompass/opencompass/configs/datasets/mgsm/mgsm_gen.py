from mmengine.config import read_base

with read_base():
    # from .mgsm_gen_d967bc import mgsm_datasets
    # from .mgsm_gen_v2 import mgsm_datasets
    from .mgsm_gen_fewshot_v1 import mgsm_datasets

