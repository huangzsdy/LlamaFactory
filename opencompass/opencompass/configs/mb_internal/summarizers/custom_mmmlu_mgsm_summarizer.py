from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.mmmlu import mmmlu_summary_groups
    from ...summarizers.groups.mgsm import mgsm_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ..summarizers.groups.cmmlu_cot import cmmlu_cot_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mmmlu
        ['mmmlu', 'naive_average'],

        # mgsm
        ['mgsm', 'accuracy'],


        # cmmlu
        ['cmmlu', 'naive_average'],
        ['cmmlu_cot', 'naive_average'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
