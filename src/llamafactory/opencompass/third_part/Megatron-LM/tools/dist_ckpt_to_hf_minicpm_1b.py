"""Pretrain GPT."""
import sys
sys.path.append("./")
import os
import torch
from functools import partial

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.modelbest_sdk_dataset_builder import ModelBestSDKDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_mla_spec,
)


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.use_mla:
                transformer_layer_spec = get_gpt_layer_with_mla_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size if not args.use_modelbest_sdk else args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base
        )

    return model


if __name__ == "__main__":

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'Llama2Tokenizer'})
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, 
                        model_type=ModelType.encoder_or_decoder)
    args = get_args()
    model_real = model[0]

    # extract megatron state dict
    state_dict_megatron = model_real.state_dict()
    new_sd = dict()
    for k in state_dict_megatron:
        if "_extra" in k:
            continue
        new_sd[k] = state_dict_megatron[k]
        print(k, state_dict_megatron[k].shape)

    # torch.save({"model": new_sd}, args.save)
    # param name mapping
    state_dict_hf = dict()
    state_dict_hf["model.embed_tokens.weight"] = state_dict_megatron["embedding.word_embeddings.weight"]
    state_dict_hf["model.norm.weight"] = state_dict_megatron["decoder.final_layernorm.weight"]

    assert args.num_attention_heads % args.num_query_groups == 0
    assert args.hidden_size % args.num_attention_heads == 0

    num_query_heads_per_group = args.num_attention_heads // args.num_query_groups

    for layer_idx in range(args.num_layers):
        state_dict_hf[f"model.layers.{layer_idx}.input_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"]
        qkv_proj = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
        # if args.group_query_attention:
        #     q_proj, k_proj, v_proj = torch.split(qkv_proj, split_size_or_sections=[args.kv_channels * args.num_attention_heads, args.kv_channels * args.num_query_groups, args.kv_channels * args.num_query_groups], dim=0)
        # else:
        #     q_proj, k_proj, v_proj = torch.split(qkv_proj, split_size_or_sections=args.kv_channels * args.num_attention_heads, dim=0)

        qkv_proj_split = torch.split(qkv_proj, split_size_or_sections=(args.hidden_size // args.num_attention_heads), dim=0)
        
        q_proj_list, k_proj_list, v_proj_list = [], [], []
        for i in range(args.num_query_groups):
            q_proj_list.extend(qkv_proj_split[(num_query_heads_per_group + 2) * i: (num_query_heads_per_group + 2) * i + num_query_heads_per_group])
            k_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
            v_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])
        
        q_proj = torch.cat(q_proj_list, dim=0)
        k_proj = torch.cat(k_proj_list, dim=0)
        v_proj = torch.cat(v_proj_list, dim=0)
        
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"]
        state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"]

        linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
        gate_proj, up_proj = torch.split(linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
        state_dict_hf[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
        state_dict_hf[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
        state_dict_hf[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"]

    # save and generate hf repository
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    torch.save(state_dict_hf, os.path.join(args.save, "pytorch_model.bin"))