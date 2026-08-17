"""
多数据集评测脚本
使用Qwen2.5-7B-Instruct模型对多个jsonl数据集进行评测
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径 - 使用本地模型
model_path = "/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/saves/qwen2.5-7b/full/cpt"

# 数据集路径
datasets_dir = "/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass/datasets"

# 输出目录
output_dir = "/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass/outputs"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading model from {model_path}...")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    padding_side='left'
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully!\n")

def evaluate_code_generation(data):
    """评测代码生成任务"""
    prompt = data['prompt']
    messages = [{"role": "user", "content": f"Write Python code for the following problem:\n{prompt}"}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def evaluate_knowledge(data):
    """评测知识理解任务（选择题）"""
    question = data['question']
    messages = [{"role": "user", "content": question}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return response

def evaluate_logic(data):
    """评测逻辑推理任务"""
    question = data['question']
    messages = [{"role": "user", "content": f"Solve this logical expression: {question}"}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return response

def evaluate_long_context(data):
    """评测长程依赖任务（阅读理解）"""
    context = data['context']
    question = data['input']
    messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the context."}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return response

# 评测配置
dataset_configs = [
    {
        "name": "代码生成",
        "file": "代码生成.jsonl",
        "eval_func": evaluate_code_generation,
        "output_key": None,  # 代码生成没有标准答案
    },
    {
        "name": "知识理解",
        "file": "知识理解.jsonl",
        "eval_func": evaluate_knowledge,
        "output_key": "answer",
    },
    {
        "name": "逻辑推理",
        "file": "逻辑推理.jsonl",
        "eval_func": evaluate_logic,
        "output_key": "answer",
    },
    {
        "name": "长程依赖",
        "file": "长程依赖.jsonl",
        "eval_func": evaluate_long_context,
        "output_key": "answers",
    },
]

# 执行评测
for config in dataset_configs:
    dataset_name = config["name"]
    dataset_file = config["file"]
    eval_func = config["eval_func"]
    output_key = config["output_key"]
    
    print(f"\n{'='*60}")
    print(f"评测数据集: {dataset_name}")
    print(f"{'='*60}")
    
    filepath = os.path.join(datasets_dir, dataset_file)
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        continue
    
    # 读取数据
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    correct = 0
    total = 0
    
    for i, line in enumerate(lines):
        if i >= 5:  # 每数据集评测5个样本
            break
            
        data = json.loads(line.strip())
        
        print(f"\n[{i+1}] 处理中...")
        
        try:
            prediction = eval_func(data)
            
            # 如果有标准答案，计算准确率
            if output_key and output_key in data:
                ground_truth = str(data[output_key]).strip().upper()
                pred_str = prediction.strip().upper()
                
                # 对于选择题，检查是否包含正确答案
                is_correct = ground_truth in pred_str or pred_str in ground_truth
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": data.get("question", data.get("prompt", data.get("input", "")))[:100],
                    "ground_truth": data[output_key],
                    "prediction": prediction,
                    "correct": is_correct
                })
                
                print(f"  问题: {results[-1]['question']}")
                print(f"  标准答案: {data[output_key]}")
                print(f"  预测: {prediction[:100]}...")
                print(f"  正确: {is_correct}")
            else:
                results.append({
                    "prompt": data.get("prompt", ""),
                    "prediction": prediction,
                })
                print(f"  Prompt: {data.get('prompt', '')[:100]}...")
                print(f"  预测: {prediction[:100]}...")
                
        except Exception as e:
            print(f"  错误: {e}")
    
    # 输出统计
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")

print("\n" + "="*60)
print("所有数据集评测完成！")
print("="*60)
