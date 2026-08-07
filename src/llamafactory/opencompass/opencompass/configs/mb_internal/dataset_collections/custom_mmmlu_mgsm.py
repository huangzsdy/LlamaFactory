from mmengine.config import read_base

with read_base():
    from ...datasets.mmmlu.mmmlu_ppl_c51a84_new import mmmlu_datasets, mmmlu_datasets_map
    from ...datasets.mgsm.mgsm_gen_d967bc import mgsm_datasets, mgsm_datasets_map
    # from ...datasets.mgsm.mgsm_gen_fewshot_v1 import mgsm_datasets, mgsm_datasets_map

hf_infer_datasets_total = []

def assert_no_overlap(*dicts):
    # ensure that no single key exists in different dicts
    assert len(set().union(*(d.keys() for d in dicts))) == sum(len(d) for d in dicts)

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = [value]

    return result

base_datasets_total = mmmlu_datasets + mgsm_datasets
datasets_map = merge_dicts(mmmlu_datasets_map, mgsm_datasets_map)

# base_datasets_total = mgsm_datasets
# datasets_map = merge_dicts(mgsm_datasets_map)

#datasets_map = merge_dicts(mmmlu_datasets_map, mgsm_datasets_map)
print(f"{__file__}, {datasets_map.keys()}")

# base_datasets_total = mmmlu_datasets
# datasets_map = merge_dicts(mmmlu_datasets_map)
