# 主训练阶段数据 (text 65% / wiki 15% / qa_ctx 17% / pure_qa 3%)
python scripts/convert_cpt_data.py \
    --input your_raw_data.jsonl \
    --output data/cpt_dataset/cpt_main.jsonl \
    --ratio-a 0.65 --ratio-b 0.15 --ratio-c 0.17 --ratio-d 0.03

# 退火阶段数据 (text 50% / wiki 25% / qa_ctx 25%)
python scripts/convert_cpt_data.py \
    --input your_raw_data.jsonl \
    --output data/cpt_dataset/cpt_anneal.jsonl \
    --ratio-a 0.50 --ratio-b 0.25 --ratio-c 0.25 --ratio-d 0.00


llamafactory-cli train examples/train_full/qwen2.5_7b_full_cpt.yaml


