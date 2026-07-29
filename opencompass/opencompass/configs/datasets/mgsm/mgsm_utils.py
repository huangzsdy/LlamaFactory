import torch
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import ICL_INFERNECERS
from opencompass.utils import get_logger

@ICL_INFERNECERS.register_module()
class DebugGenInferencer(GenInferencer):
    """调试用的推理器，保存原始模型输出"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_model_outputs = []
        self.logger = get_logger(log_level='DEBUG')


    def generate(self, inputs, **kwargs):
        # 调用父类生成
        results = super().generate(inputs, **kwargs)
        import time
        # 保存原始输出
        for i, (input_text, result) in enumerate(zip(inputs, results)):
            raw_model_output = {
                'index': i,
                'input_prompt': input_text,  # 完整的提示词
                'raw_model_output': result,  # 模型的完整响应
                'timestamp': time.time()
            }
            self.raw_model_outputs.append(raw_model_output)
            self.logger.debug(f"raw_model_output:{raw_model_output}")

        return results

    def predict(self, data, **kwargs):
        self.logger.debug("\n" + "=" * 60)
        self.logger.debug("[DebugGenInferencer] predict() 被调用")
        self.logger.debug(f"输入数据数量:{len(data)}")
        self.logger.debug("=" * 60)

        inputs = self.get_inputs(data)
        with torch.no_grad():
            # 调用父类生成
            results = self.model.generate(inputs, **kwargs)

            import time
            # 保存原始输出
            for i, (input_text, result) in enumerate(zip(inputs, results)):
                raw_model_output = {
                    'index': i,
                    'input_prompt': input_text,  # 完整的提示词
                    'raw_model_output': result,  # 模型的完整响应
                    'timestamp': time.time()
                }
                self.raw_model_outputs.append(raw_model_output)
                self.logger.debug(f"raw_model_output:{raw_model_output}")

            return results

    def save_raw_outputs(self, filename='model_raw_outputs.json'):
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.raw_model_outputs, f, ensure_ascii=False, indent=2)