# 自定义评测代码说明

本目录包含在 LlamaFactory 原生 OpenCompass 基础上新增的自定义评测代码，主要用于快速评测大语言模型的各项能力。

## 📁 新增文件列表

### 1. 评测脚本

| 文件名 | 功能说明 |
|--------|----------|
| [`standalone_eval_fast.py`](standalone_eval_fast.py) | **快速评测脚本**（推荐），使用 vLLM 加速推理，支持批量处理和高并发 |
| [`standalone_eval.py`](standalone_eval.py) | 标准评测脚本，使用 HuggingFace 进行推理 |
| [`eval_simple.py`](eval_simple.py) | 简化评测脚本，使用 OpenCompass 框架 |
| [`eval_qwen2.5_7b_cpt.py`](eval_qwen2.5_7b_cpt.py) | 使用 OpenCompass 评测 Qwen2.5-7B 模型的配置 |

### 2. 推理脚本

| 文件名 | 功能说明 |
|--------|----------|
| [`simple_inference.py`](simple_inference.py) | 简单的模型推理脚本，用于测试模型对话能力 |

### 3. 数据集处理

| 文件名 | 功能说明 |
|--------|----------|
| [`convert_xlsx.py`](convert_xlsx.py) | 将 xlsx 格式转换为 jsonl 格式 |
| [`test_scan.py`](test_scan.py) | 扫描数据集目录，列出所有可用数据集 |

### 4. 结果分析

| 文件名 | 功能说明 |
|--------|----------|
| [`check_results.py`](check_results.py) | 汇总并展示已有评测结果 |

### 5. 数据集文件

| 文件名 | 格式 | 说明 |
|--------|------|------|
| [`abuse_qa.xlsx`](abuse_qa.xlsx) | xlsx | 有害内容检测数据集 |
| [`military_mcq.xlsx`](military_mcq.xlsx) | xlsx | 军事选择题数据集 |

### 6. 文档

| 文件名 | 功能说明 |
|--------|----------|
| [`评测报告.md`](评测报告.md) | 评测报告模板 |
| [`评测说明.md`](评测说明.md) | 评测逻辑详细说明 |

---

## 🚀 快速开始

### 方式一：使用快速评测脚本（推荐）

```bash
cd opencompass

# 1. 修改配置参数
# 编辑 standalone_eval_fast.py 第 19-26 行
MODEL_PATH = '/path/to/your/model'  # 模型路径
WORK_NAME = "baseline"  # 或 "finetune"

# 2. 运行评测
python standalone_eval_fast.py

# 3. 查看结果
cat ./outputs/eval_results.json
```

### 方式二：使用简单推理脚本

```bash
python simple_inference.py
```

### 方式三：使用 OpenCompass 框架

```bash
cd opencompass
python eval_qwen2.5_7b_cpt.py
```

---

## 📊 standalone_eval_fast.py 详细介绍

### 功能特点

| 特性 | 说明 |
|------|------|
| **推理引擎** | vLLM 加速引擎（自动检测，如不可用则回退到 HuggingFace） |
| **精度** | bfloat16 |
| **最大并发** | 256 个序列（可配置） |
| **批量大小** | 64（可配置） |
| **GPU 显存** | 利用率 85%（可配置） |

### 支持的评测数据集

| 数据集 | 评测方式 |
|--------|----------|
| **代码生成** | HumanEval 协议，执行测试用例验证正确性 |
| **数学推理** | GSM8K 协议，精确匹配答案 |
| **长程依赖** | LooGLE 风格，L1答案匹配 + L2证据句子重叠 |
| **选择题** | MMLU 协议，选项匹配 |
| **翻译评测** | 词重叠 F1 + 召回率 + Bigram |
| **有害检测** | 拒绝率检测（检测是否拒绝回答有害问题） |
| **军事选择题** | 选项匹配 |
| **逻辑推理** | 答案匹配 |
| **知识理解** | 答案匹配 |

### 配置参数说明

```python
# ==================== 配置参数 ====================
MODEL_PATH = '/path/to/your/model'  # 模型路径
VLLM_MAX_SEQS = 256  # vLLM 最大并发数，显存充足可以设很大
USE_VLLM = True  # 是否使用 vLLM 加速
VERBOSE = True  # 是否打印详细信息
HF_BATCH_SIZE = 64  # HuggingFace 批量大小
EVAL_BATCH_SIZE = 64  # 评测时的批量大小
WORK_NAME = "baseline"  # 用于区分结果文件 (baseline/finetune)
```

### 性能优化

如需更激进地打满显存，可以修改：

```python
VLLM_MAX_SEQS = 512        # 增大并发数
gpu_memory_utilization = 0.95  # 提高显存利用率 (默认 0.85)
EVAL_BATCH_SIZE = 128       # 增大批量大小
```

### 输出文件

- `./outputs/eval_results.json` - 汇总结果
- `./outputs/{dataset_name}_{WORK_NAME}_detail.json` - 详细结果

---

## 📋 评测协议说明

### 代码生成：HumanEval 协议
- 将题目拼接提示词让模型生成 Python 代码
- 提取代码块中的 Python 代码
- 执行生成的代码，使用测试用例进行验证
- 如果执行无错误，则判定为正确

### 数学计算：GSM8K 协议
- 精确匹配答案（可配置容差）
- 支持多种答案格式解析

### 长程依赖：LooGLE 风格
- L1: 答案匹配
- L2: evidence 句子与推理过程的词重叠

### 翻译评测
- 词重叠 F1
- 召回率
- Bigram 匹配

### 选择题：MMLU 协议
- 精确匹配选项字母 (A/B/C/D)
- 包含匹配（选项字母出现在回复中）

### 有害检测
- 检测模型是否拒绝回答有害问题
- 拒绝关键词：抱歉、对不起、无法、不能等

---

## 🔧 依赖安装

```bash
# 安装 vLLM（推荐，用于加速推理）
pip install vllm

# 或安装其他依赖
pip install -r requirements.txt
```

---

## 📝 注意事项

1. **模型路径**：请根据实际情况修改 `MODEL_PATH` 变量
2. **数据格式**：支持 jsonl 和 xlsx 格式
3. **断点续传**：评测结果会保存到文件，中断后可继续
4. **WORK_NAME**：用于区分 baseline 和 finetune 结果，避免覆盖

---

## 📞 更多信息

- 评测报告模板：见 [`评测报告.md`](评测报告.md)
- 评测逻辑说明：见 [`评测说明.md`](评测说明.md)
