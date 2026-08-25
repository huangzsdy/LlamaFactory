# # ========================================
# # 数据混合方案：领域数据 + Fineweb
# # ========================================
# # 混合比例: 60% 领域数据 + 40% Fineweb (有效防止灾难性遗忘)

# # 使用mix_cpt_data.py脚本混合数据
# python scripts/mix_cpt_data.py \
#     --domain-data data/cpt_dataset/your_domain_data.jsonl \
#     --fineweb-data data/cpt_dataset/fineweb_processed.jsonl \
#     --output data/cpt_dataset/cpt_main.jsonl \
#     --domain-ratio 0.6 \
#     --fineweb-ratio 0.4

# # 可选：限制最大样本数
# # python scripts/mix_cpt_data.py \
# #     --domain-data data/cpt_dataset/your_domain_data.jsonl \
# #     --fineweb-data data/cpt_dataset/fineweb_processed.jsonl \
# #     --output data/cpt_dataset/cpt_main.jsonl \
# #     --domain-ratio 0.6 \
# #     --fineweb-ratio 0.4 \
# #     --max-samples 100000

# # 退火阶段可以提高新领域数据比例 (70% 领域 + 30% Fineweb)
# # python scripts/mix_cpt_data.py \
# #     --domain-data data/cpt_dataset/your_domain_data.jsonl \
# #     --fineweb-data data/cpt_dataset/fineweb_processed.jsonl \
# #     --output data/cpt_dataset/cpt_anneal.jsonl \
# #     --domain-ratio 0.7 \
# #     --fineweb-ratio 0.3


# llamafactory-cli train examples/train_full/qwen2.5_7b_full_cpt.yaml


