from mmengine.config import read_base

with read_base():
    # from .mmmlu_ppl_c51a84 import mmmlu_datasets  # noqa: F401, F403
    # from .mmmlu_ppl_v2 import mmmlu_datasets  # noqa: F401, F403
    from .mmmlu_ppl_fewshot_v1 import mmmlu_datasets  # noqa: F401, F403


