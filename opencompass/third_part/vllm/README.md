# MiniCPM LongRoPE 手动更改 VLLM 介绍

**NOTE**: 当前支持的 VLLM 版本为 0.6.3.post1，其他的本本需要自行测试。

因为 MiniCPM (`MiniCPMForCausalLM`) 不支持 LongRoPE，运行过程中会报错，所以需要手动将一些文件加入到 VLLM 中。

## Huggingface model config

保存到 HF 模型 `config.json` 中，`architectures` 需要从 `MiniCPMForCausalLM` 更改为 `MiniCPMLongRopeForCausalLM`.

全量 `config.json` 文件内容如下：

```json
{
    "_name_or_path": "openbmb/CPM-2B",
    "architectures": [
        "MiniCPMLongRopeForCausalLM"
    ],
    "auto_map": {
        "AutoConfig": "configuration_minicpm.MiniCPMConfig",
        "AutoModel": "modeling_minicpm.MiniCPMModel",
        "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
        "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification"
    },
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.1,
    "intermediate_size": 3840,
    "max_position_embeddings": 32768,
    "num_attention_heads": 24,
    "num_hidden_layers": 52,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "type": "su",
        "long_factor": [1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297],
        "short_factor": [1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297],
        "original_max_position_embeddings": 32768
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.36.0",
    "use_cache": true,
    "vocab_size": 73448,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4
}
```

## 将必要文件复制到 VLLM 中

1. 查看 `vllm` 的安装路径

```bash
pip show vllm
```

获取到 `Location` 信息，如 `/home/miniconda3/envs/opencompass/lib/python3.10/site-packages`, 因此 `vllm` 的安装路径为 `/home/miniconda3/envs/opencompass/lib/python3.10/site-packages/vllm`.

2. 复制必要文件到 `vllm` 下面。

```bash
VLLM_PATH="/home/miniconda3/envs/opencompass/lib/python3.10/site-packages/vllm"

# 将 minicpm_longrope 加入到 vllm models 中
cp third_part/vllm/models/minicpm_longrope.py $VLLM_PATH/model_executor/models/.
# 更新 registry.py， 加入 minicpm_longrope
cp third_part/vllm/models/registry.py $VLLM_PATH/model_executor/models/.
```
