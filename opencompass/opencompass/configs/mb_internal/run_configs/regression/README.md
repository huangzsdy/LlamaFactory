# 回归测试

每一个内部版本应该做一次回归测试，来保证模型推理结果没有异常。选择进行回归测试的模型包括：

- minicpm-1b 4k 基座模型：`/user/zhangyixuan/models/opencompass_regression/jobid_2542-iter_5000`
- minicpm-1b 32k sft 模型：`/user/zhangyixuan/models/opencompass_regression/jobid_6860-iter_8330`

## 启动回归测试

1. 准备 OpenCompass 环境
   - 需要注意，minicpm-1b 32k sft 模型可能需要更改 vllm 文件，具体操作请参考 [vllm README](../../../../third_part/vllm/README.md)。
   - third_part/vllm/README.md
2. 运行回归测试脚本

```bash
bash configs/mb_internal/run_configs/regression/regression.sh
```

## 结果对比

一般情况下，波动 0.3 以内的结果都认为是正常波动。如果有异常情况，需要根据 prediction 进行对比分析。

### minicpm-1b 4k 基座模型

#### 英文评测集

| dataset | version | metric | mode | jobid_2542@iter_5000 |
|----- | ----- | ----- | ----- | -----|
| mmlu | - | naive_average | ppl | 47.49 |
| commonsense_qa | e51e32 | accuracy | ppl | 57.49 |
| hellaswag | 47bff9 | accuracy | ppl | 55.73 |
| ARC-c | a450bd | accuracy | ppl | 38.98 |
| ARC-e | a450bd | accuracy | ppl | 59.79 |
| piqa | 1cf9f0 | accuracy | ppl | 73.50 |
| siqa | ced5f6 | accuracy | ppl | 42.43 |
| winogrande | 55a66e | accuracy | ppl | 54.14 |
| openbookqa_fact | da3815 | accuracy | ppl | 71.00 |
| bbh | - | naive_average | gen | 35.04 |
| GPQA_diamond | 6bf57a | accuracy | ppl | 22.73 |
| math | db136b | accuracy | gen | 11.00 |
| gsm8k | 17d0dc | accuracy | gen | 29.95 |
| sanitized_mbpp | 742f0c | score | gen | 41.25 |
| openai_humaneval | d2537e | humaneval_pass@1 | gen | 26.22 |

#### 中文评测集

| dataset | version | metric | mode | jobid_2542@iter_5000 |
|----- | ----- | ----- | ----- | -----|
| cmmlu | - | naive_average | ppl | 48.89 |
| ceval | - | naive_average | ppl | 48.84 |
| commonsenseqa_cn | 971f48 | accuracy | ppl | 38.66 |
| csl_dev | 46f772 | accuracy | ppl | 61.25 |
| csl_test | 46f772 | accuracy | ppl | 61.35 |
| nq_cn | 53d3dd | score | gen | 0.83 |
| ocnli | c4cb6c | accuracy | gen | 34.07 |
| cmnli | 1abf97 | accuracy | gen | 33.20 |
| chid-dev | 90451d | accuracy | ppl | 75.74 |
| chid-test | 90451d | accuracy | ppl | 77.57 |
| C3 | e24a31 | accuracy | ppl | 54.79 |

### minicpm-1b 32k sft 模型

#### 核心评测集

| dataset | version | metric | mode | jobid_6860@iter_8330 |
|----- | ----- | ----- | ----- | -----|
| mmlu | - | naive_average | gen | 62.14 |
| mmlu_cot | - | naive_average | gen | 59.60 |
| cmmlu | - | naive_average | gen | 58.53 |
| ceval | - | naive_average | gen | 60.65 |
| ARC-c | 1e0de5 | accuracy | gen | 77.63 |
| ARC-e | 1e0de5 | accuracy | gen | 85.54 |
| bbh | - | naive_average | gen | 48.54 |
| GPQA_diamond | 4baadb | accuracy | gen | 23.74 |
| math | 393424 | accuracy | gen | 34.70 |
| gsm8k | 1d7fe4 | accuracy | gen | 66.94 |
| sanitized_mbpp | a447ff | score | gen | 52.14 |
| openai_humaneval | 8e312c | humaneval_pass@1 | gen | 51.83 |
