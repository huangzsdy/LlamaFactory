"""
SacreBleuEvaluator for OpenCompass.

评测翻译生成结果，使用 sacrebleu 计算 BLEU 和 chrF。
安装：pip install sacrebleu
"""

import sacrebleu
from opencompass.registry import EVALUATORS
from opencompass.utils import get_logger

logger = get_logger(__name__)


@EVALUATORS.register_module()
class SacreBleuEvaluator:
    """翻译评测器：BLEU + chrF。"""

    def __init__(self, metrics: list = None):
        """
        Args:
            metrics: 要返回的指标列表，可选 'bleu', 'chrf'
        """
        if metrics is None:
            metrics = ['bleu', 'chrf']
        self.metrics = metrics

    def score(self, predictions: list, references: list) -> dict:
        """
        Args:
            predictions: list of str, 模型生成的翻译
            references: list of str (或 list of list of str), 参考答案
        Returns:
            dict of metric_name -> score
        """
        # 统一 references 格式
        refs = []
        for r in references:
            if isinstance(r, list):
                refs.append(r[0])  # 取第一个参考答案
            else:
                refs.append(r)

        results = {}

        if 'bleu' in self.metrics:
            bleu = sacrebleu.corpus_bleu(predictions, [refs])
            results['bleu'] = round(bleu.score, 4)

        if 'chrf' in self.metrics:
            chrf = sacrebleu.corpus_chrf(predictions, refs)
            results['chrf'] = round(chrf.score, 4)

        return results
