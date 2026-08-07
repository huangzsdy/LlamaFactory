# 项目自定义修改说明

本文档记录了对 LlamaFactory 项目的所有自定义修改和扩展功能。

---

## 1. OpenCompass 评测集成

### 功能说明
集成了 OpenCompass 评测框架，支持对训练后的模型进行全面的性能评估。

### 支持的评测能力
- 基础能力评测（知识理解、语言理解、逻辑推理、数学计算、代码生成等）
- 中文评测集支持
- 自定义评测数据集

### 使用方法
```bash
# 运行评测
cd opencompass
bash work.sh
```

---

## 2. 数据格式处理

### 2.1 XLSX 文件处理

#### 功能说明
支持将 Excel 文件（.xlsx）转换为模型训练可用的 JSONL 格式。

#### 支持字段
- `question`: 问题/题目
- `answer`: 答案
- `passage`: 相关段落（可选）
- `target_scores`: 选项分数（可选，用于选择题）

#### 转换命令
```bash
python datasets/eval_all.py --input datasets/your_file.xlsx --output datasets/output.jsonl
```

### 2.2 文本数据混合处理

#### 功能说明
支持对不同类型的数据进行配比混合，用于 CPT（继续预训练）训练。

#### 数据类型
1. **纯文本 (text)**: 原始文本内容，占比 60-65%
2. **Wikipedia 改写 (wiki)**: Wikipedia 风格的文本改写，占比 15-25%
3. **QA with Context (qa_ctx)**: 基于上下文的问答，占比 17-25%
4. **Pure QA (pure_qa)**: 纯问答对，占比 2-3%

#### 暴露度控制
- 支持设置最大曝光次数（max_exposure）
- 避免单一数据过度重复训练

#### 使用方法
```bash
python scripts/convert_cpt_data.py \
    --input_dir /path/to/input.jsonl \
    --output_dir data/cpt_dataset \
    --max_exposure 3
```

---

## 3. CPT（继续预训练）配置

### 3.1 训练配置文件

#### 主训练阶段配置
位置: `examples/train_full/qwen2.5_7b_full_cpt.yaml`

关键参数:
```yaml
### method
stage: pt                         # PT (预训练/继续预训练)
finetuning_type: full             # 全参数训练
deepspeed: examples/deepspeed/ds_z2_config.json

### dataset
dataset_dir: data/cpt_dataset
dataset: cpt_main,cpt_anneal

### train
max_steps: 5000                  # 主训练阶段步数
learning_rate: 3.0e-5           # 学习率
lr_scheduler_type: cosine         # 学习率调度
bf16: true                       # 精度
```

### 3.2 分阶段训练

#### 第一阶段：主训练（5000步）
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_full/qwen2.5_7b_full_cpt.yaml
```

#### 第二阶段：退火（600步）
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_full/qwen2.5_7b_full_cpt.yaml \
    dataset=code_anneal \
    resume_from_checkpoint=saves/qwen2.5-7b/full/cpt/checkpoint-5000 \
    max_steps=600 \
    learning_rate=1e-6
```

### 3.3 Checkpoint 配置

为减少保存的 checkpoint 数量，配置如下：
```yaml
save_steps: 2000                  # 每2000步保存一次
save_total_limit: 1              # 只保留1个最新checkpoint
save_strategy: steps              # 按步数保存
save_only_model: false           # 保存完整模型权重
```

---

## 4. 数据集配置格式

### 4.1 dataset_info.json 格式

```json
{
  "dataset_name": {
    "file_name": "data.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "text"
    },
    "description": "数据集描述"
  }
}
```

### 4.2 数据字段说明

| 字段名 | 说明 | 是否必需 |
|--------|------|----------|
| text | 原始文本内容 | 是 |
| instruction | 指令/问题 | 是 |
| output | 回答/答案 | 否 |
| prompt | 提示词 | 是 |

---

## 5. 训练参数参考

### 5.1 全参数训练配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| per_device_train_batch_size | 4 | 单卡 batch size |
| gradient_accumulation_steps | 32 | 梯度累积步数 |
| learning_rate | 3e-5 | 初始学习率 |
| max_steps | 5000 | 训练步数 |
| warmup_ratio | 0.05 | 预热比例 |
| bf16 | true | 使用 bf16 精度 |

### 5.2 学习率调度

推荐使用 cosine 调度器，可自动实现学习率退火效果。

---

## 6. 常见问题

### Q1: 如何指定 GPU 训练？
```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_full/qwen2.5_7b_full_cpt.yaml
```

### Q2: 如何从 checkpoint 继续训练？
使用 `resume_from_checkpoint` 参数指定 checkpoint 路径。

### Q3: 数据量不足怎么办？
确保数据集足够大，一般建议至少 10 万条以上。

---

## 7. 文件结构

```
LlamaFactory/
├── examples/
│   └── train_full/
│       └── qwen2.5_7b_full_cpt.yaml    # CPT 训练配置
├── scripts/
│   └── convert_cpt_data.py               # CPT 数据转换脚本
├── data/
│   ├── cpt_dataset/                     # CPT 数据集目录
│   │   └── dataset_info.json
│   └── mixed_train/                     # 混合训练数据
├── datasets/
│   ├── eval_all.py                      # XLSX 转换脚本
│   └── *.xlsx                          # 原始数据
├── opencompass/                          # 评测框架
│   └── work.sh                          # 评测脚本
└── docs/
    └── CUSTOM_MODIFICATIONS.md           # 本文档
```

---

## 8. 更新日志

### 2024-xx-xx
- 添加 OpenCompass 评测框架
- 添加 XLSX 文件处理支持
- 添加 CPT 数据混合处理功能
- 添加分阶段训练配置
- 添加 Checkpoint 优化配置
