#!/usr/bin/env python3
"""
快速评测脚本 - 优化版本
使用 vLLM 加速推理
"""

import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置工作目录
# os.chdir('/mnt/c/Users/ThinkPad/my_own_files/study/llm/LlamaFactory-main/LlamaFactory-main/opencompass')

# ==================== 显存优化选项 ====================
# 如果显存不够，修改以下参数：

# transformers 参数 (当前使用的是 transformers)
HF_BATCH_SIZE = 1  # 【修改】将批量大小减小为 1，大幅降低激活值显存占用
USE_CPU_OFFLOAD = False  # 设为 True 可启用 CPU 卸载 (模型在CPU，推理时加载到GPU)
USE_QUANTIZATION = True   # 【修改】设为 True 开启 8-bit 量化，显存直接减半（需安装 bitsandbytes）

# vLLM 参数 (如果没有安装 vllm，则使用 transformers)
VLLM_MAX_SEQS = 16        # 【修改】调小 vLLM 最大并发序列数，降低 KV Cache 显存

# ====================================================

# 尝试导入 vLLM，如果不可用则使用 transformers
try:
    from vllm import LLM, SamplingParams
    USE_VLLM = True
    print("使用 vLLM 加速")
except ImportError:
    # 预先导入 Qwen2 模型类，避免 AutoModel 映射问题
    try:
        from transformers import Qwen2ForCausalLM
        print("Qwen2ForCausalLM 已预加载")
    except ImportError:
        print("警告: 无法预加载 Qwen2ForCausalLM")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_VLLM = False
    print("使用 transformers")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as err:  # 修复原代码中 Python 3 的异常捕获语法错误
                    print(f"Exception:{err}, line:{line}")

    return data

def load_data(file_path):
    if file_path.endswith('.jsonl'):
        return load_jsonl(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path).to_dict('records')
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

def load_model():
    model_path = '/mnt/c/Users/ThinkPad/Downloads/copy_models/Qwen/Qwen2.5-7B-Instruct'
    print(f"正在加载模型: {model_path}")
    
    if USE_VLLM:
        # 使用 vLLM - 极快
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,  # 如果有多个GPU可以增加
            dtype="half",  # 半精度
            max_num_seqs=VLLM_MAX_SEQS,  # 批量大小
            gpu_memory_utilization=0.8,  # 【修改】稍微调低显存利用率上限，防止溢出
        )
        return llm, None, 'vllm'
    else:
        # 使用 transformers
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("使用CPU")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        
        # 尝试使用 Qwen2ForCausalLM 直接加载
        try:
            from transformers import Qwen2ForCausalLM
            print("尝试直接加载 Qwen2ForCausalLM...")
            
            # 【修改】如果开启了量化，直接加载时也应用量化配置
            if USE_QUANTIZATION:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = Qwen2ForCausalLM.from_pretrained(
                    model_path, device_map='auto', trust_remote_code=True, quantization_config=quantization_config
                )
            else:
                model = Qwen2ForCausalLM.from_pretrained(
                    model_path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16
                )
            print("直接加载成功!")
            return model, tokenizer, device
        except Exception as e:
            print(f"直接加载失败: {e}")
            print("尝试使用 AutoModelForCausalLM...")
        
        # 根据配置选择加载方式
        if USE_QUANTIZATION:
            # 需要安装 bitsandbytes: pip install bitsandbytes
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map='auto', 
                trust_remote_code=True, 
                quantization_config=quantization_config
            )
        elif USE_CPU_OFFLOAD:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='cpu', trust_remote_code=True, torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        
        return model, tokenizer, device

def generate_vllm(llm, prompts, max_new_tokens=64):
    """使用 vLLM 生成 - 极快"""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop=None,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

def generate_hf(model, tokenizer, prompts, max_new_tokens=64, device='cuda'):
    """使用 transformers 生成 - 支持批量处理"""
    import torch
    
    # 批量处理
    results = []
    batch_size = HF_BATCH_SIZE
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results

def generate_code_hf(model, tokenizer, prompt, max_new_tokens=512, device='cuda'):
    """使用 transformers 生成代码 - 增加长度限制"""
    import torch
    full_prompt = f"请只给出Python代码，不要解释: {prompt}\n\n```python\n"
    
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if '```python' in response:
        response = response.split('```python')[-1]
    if '```' in response:
        response = response.split('```')[0]
    
    code = response.strip()
    
    # 检测并修复被截断的代码 - 增强版
    open_parens = code.count('(')
    close_parens = code.count(')')
    open_brackets = code.count('[')
    close_brackets = code.count(']')
    open_braces = code.count('{')
    close_braces = code.count('}')
    
    is_truncated = (open_parens > close_parens or 
                    open_brackets > close_brackets or 
                    open_braces > close_braces)
    
    if is_truncated:
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            # 跳过不完整的行
            if stripped.endswith(('(', '[', '{', ',', '\\', '+', '-', '*', '/', '=', '->', ':')):
                continue
            # 跳过只有缩进的空行和注释
            if stripped == '' or stripped.startswith('#'):
                continue
            fixed_lines.append(line)
        code = '\n'.join(fixed_lines)
        
        # 再次检查并尝试修复函数定义不完整的情况
        if 'def ' in code and ':' not in code.split('def ')[-1][:100]:
            # 移除不完整的函数定义
            parts = code.split('def ')
            code = 'def '.join(parts[:-1])
            if len(parts) > 1:
                code += '\n    pass'
    
    return code

def execute_code_test(generated_code, test_code):
    """执行代码测试"""
    import re
    try:
        code = generated_code.strip()
        
        # 预处理：移除不完整行
        lines = code.split('\n')
        valid_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(('(', '[', '{', ',', '\\', '+', '-', '*', '/', '=')):
                continue
            if stripped == '' or stripped.startswith('#'):
                continue
            valid_lines.append(line)
        code = '\n'.join(valid_lines)
        
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not func_match:
            return False, "未找到函数定义"
        
        func_name = func_match.group(1)
        full_code = code + "\n\n" + test_code
        
        local_vars = {}
        exec(full_code, {}, local_vars)
        
        if func_name not in local_vars:
            return False, "函数未定义"
        
        return True, "执行成功"
        
    except SyntaxError as e:
        return False, f"语法错误: {str(e)[:40]}"
    except Exception as e:
        return False, f"执行错误: {str(e)[:40]}"

def load_existing_results(dataset_name):
    """加载已存在的评测结果"""
    detail_file = f'./outputs/{dataset_name}_detail.json'
    if os.path.exists(detail_file):
        try:
            with open(detail_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_detail_results(dataset_name, results):
    """保存详细的评测结果"""
    detail_file = f'./outputs/{dataset_name}_detail.json'
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def is_correct_result(r):
    """检查结果是否正确 - 处理整数和字典两种格式"""
    if isinstance(r, bool):
        return r
    elif isinstance(r, int):
        return r > 0  # 假设 > 0 表示正确
    elif isinstance(r, dict):
        return r.get('correct', False)
    return False

def main():
    datasets_dir = './datasets'
    jsonl_files = sorted(glob.glob(os.path.join(datasets_dir, '*.jsonl')))
    xlsx_files = sorted(glob.glob(os.path.join(datasets_dir, '*.xlsx')))
    data_files = jsonl_files + xlsx_files
    
    print(f"找到 {len(data_files)} 个数据文件:")
    for f in data_files:
        print(f"  - {os.path.basename(f)}")
    
    if not data_files:
        print("未找到任何数据文件!")
        return
    
    model, tokenizer, device = load_model()
    results = {}
    
    for data_file in data_files:
        filename = os.path.basename(data_file)
        dataset_name = os.path.splitext(filename)[0]
        
        print(f"\n{'='*60}")
        print(f"正在评测: {filename}")
        
        data = load_data(data_file)
        print(f"数据条数: {len(data)}")
        
        # 加载已存在的详细结果（断点续评）
        existing_detail = load_existing_results(dataset_name)
        if existing_detail:
            print(f"找到已存在的评测结果，将跳过 {len(existing_detail)} 个已处理样本")
        
        # 检查是否所有样本都已处理完
        if len(existing_detail) >= len(data):
            print(f"所有样本已处理完成，跳过")
            # 统计已完成的准确率
            correct = sum(1 for r in existing_detail.values() if is_correct_result(r))
            accuracy = correct / len(data) * 100
            results[dataset_name] = {'correct': correct, 'total': len(data), 'accuracy': accuracy}
            print(f"结果: {correct}/{len(data)} = {accuracy:.2f}%")
            continue
        
        # 检测是否为代码生成数据集
        is_code_dataset = '代码' in dataset_name or 'code' in dataset_name.lower()
        
        if is_code_dataset:
            # 代码生成评测（优化版：先批量生成，再并行测试）
            print("检测到代码生成数据集，使用代码执行评测（并行优化）...")
            
            # 继续使用已有的 detail 结果
            detail_results = existing_detail.copy()
            correct = sum(1 for r in detail_results.values() if is_correct_result(r))
            
            # 阶段1：批量生成代码
            print("阶段1: 批量生成代码...")
            generated_items = []
            for i, item in tqdm(enumerate(data), total=len(data), desc="生成代码"):
                if str(i) in detail_results:
                    continue
                
                question = item.get('question', '') or item.get('input', '') or item.get('prompt', '')
                test_code = item.get('test_code', '') or item.get('test', '') or item.get('answer', '')
                
                # 生成代码
                if USE_VLLM:
                    prompt = f"请只给出Python代码，不要解释: {question}\n\n```python\n"
                    sampling_params = SamplingParams(temperature=0, max_tokens=512, stop=None)
                    outputs = model.generate([prompt], sampling_params)
                    generated_code = outputs[0].outputs[0].text
                    if '```python' in generated_code:
                        generated_code = generated_code.split('```python')[-1]
                    if '```' in generated_code:
                        generated_code = generated_code.split('```')[0]
                else:
                    generated_code = generate_code_hf(model, tokenizer, question, max_new_tokens=512, device=device)
                
                generated_items.append((i, question, generated_code, test_code))
            
            # 阶段2：并行执行代码测试
            if generated_items:
                print(f"阶段2: 并行执行代码测试 ({len(generated_items)} 个任务)...")
                n_workers = min(8, multiprocessing.cpu_count())
                print(f"使用 {n_workers} 个进程并行评测...")
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(execute_code_test, code, test): idx 
                              for idx, _, code, test in generated_items}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="代码执行"):
                        idx = futures[future]
                        _, question, generated_code, test_code = generated_items[[i for i, (i2, _, _, _) in enumerate(generated_items) if i2 == idx][0]]
                        
                        try:
                            is_correct, msg = future.result()
                        except Exception as e:
                            is_correct, msg = False, f"执行异常: {str(e)[:40]}"
                        
                        detail_results[str(idx)] = {
                            'question': question[:100],
                            'generated_code': generated_code[:200] if generated_code else '',
                            'test_code': test_code[:100] if test_code else '',
                            'correct': is_correct,
                            'message': msg
                        }
                        
                        if is_correct:
                            correct += 1
                        
                        # 每处理 50 个样本保存一次
                        if len(detail_results) % 50 == 0:
                            save_detail_results(dataset_name, detail_results)
            
            # 保存最终结果
            save_detail_results(dataset_name, detail_results)
            accuracy = correct / len(data) * 100
            results[dataset_name] = {'correct': correct, 'total': len(data), 'accuracy': accuracy}
            print(f"结果: {correct}/{len(data)} = {accuracy:.2f}%")
        else:
            # 普通问答评测
            # 继续使用已有的 detail 结果
            detail_results = existing_detail.copy()
            
            # 构建 prompts - 只为未处理的样本生成
            prompts = []
            prompt_indices = []
            for i, item in enumerate(data):
                if str(i) in detail_results:
                    continue
                q = item.get('question', '') or item.get('input', '') or item.get('prompt', '')
                if 'A)' in q:
                    prompts.append(f"答案: {q}")
                else:
                    prompts.append(f"答案: {q}")
                prompt_indices.append(i)
            
            # 批量生成
            if prompts:
                if USE_VLLM:
                    new_responses = generate_vllm(model, prompts, max_new_tokens=64)
                else:
                    new_responses = generate_hf(model, tokenizer, prompts, max_new_tokens=64, device=device)
                
                # 评估新生成的回复
                for idx, (i, response) in enumerate(zip(prompt_indices, new_responses)):
                    item = data[i]
                    answer = str(item.get('answer') or item.get('answers', '')).strip().upper()
                    response = response.strip().upper()
                    
                    # 提取预测答案
                    pred = None
                    if answer in 'ABCD':
                        for char in response[:50]:
                            if char in 'ABCD':
                                pred = char
                                break
                    
                    if pred is None:
                        pred = response[:20]
                    
                    # 判断
                    if answer in 'ABCD':
                        is_correct = pred == answer
                    else:
                        is_correct = answer in pred or pred in answer
                    
                    # 保存结果
                    detail_results[str(i)] = {
                        'question': item.get('question', '')[:100],
                        'answer': answer,
                        'response': response[:100],
                        'predicted': pred,
                        'correct': is_correct
                    }
            
            # 统计结果
            correct = sum(1 for r in detail_results.values() if is_correct_result(r))
            
            # 保存详细结果
            save_detail_results(dataset_name, detail_results)
            accuracy = correct / len(data) * 100
            results[dataset_name] = {'correct': correct, 'total': len(data), 'accuracy': accuracy}
            print(f"结果: {correct}/{len(data)} = {accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("评测结果汇总")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: {result['correct']}/{result['total']} = {result['accuracy']:.2f}%")
    
    Path("outputs").mkdir(exist_ok=True, parents=True)
    with open('./outputs/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: ./outputs/eval_results.json")

if __name__ == '__main__':
    main()