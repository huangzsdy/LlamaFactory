"""
简单的模型推理脚本
使用HuggingFace Transformers对Qwen2.5-7B-Instruct模型进行推理
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径 - 使用HuggingFace上的Qwen2.5-7B-Instruct
model_path = "Qwen/Qwen2.5-7B-Instruct"

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

print("Model loaded successfully!")

# 测试问题
test_questions = [
    "你好，请介绍一下你自己",
    "什么是人工智能？",
    "请写一个简单的Python程序，输出Hello World"
]

for question in test_questions:
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")
    
    # 构建prompt
    messages = [
        {"role": "user", "content": question}
    ]
    
    # 使用chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    print(f"Answer: {response}")
    print()

print("Inference completed!")
