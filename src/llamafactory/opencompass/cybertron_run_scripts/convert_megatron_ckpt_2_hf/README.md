# 将 Megatron 模型权重转换为 Hugging Face 模型权重

**Note**: 请注运行脚本时候使用的镜像需要和训练的保持一致，否则会出现错误。

## 使用方法

需要在至少有 1 GPU 的环境下运行。

### 直接路径 (direct path)

直接指定 Megatron 存储模型的路径，保存成 `${SAVE_CKPT_DIR}/${FOLDER_NAME}-iter_${CKPT_ITER}`，其中 `${FOLDER_NAME}` 为指定 Megatron 模型的文件夹名。

```bash
bash cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh \
    ${MODEL_TYPE} \
    ${SEQ_LEN} \
    ${CKPT_PATH} \
    ${CKPT_ITER} \
    ${SAVE_CKPT_DIR}
```

- `MODEL_TYPE`: 模型类型，可选择 `minicpm-1b` 或 `minicpm3-4b`。
- `SEQ_LEN`: 模型长度，可选择 `4k` 或 `32k`。
- `CKPT_PATH`: Megatron 模型存储路径。
- `CKPT_ITER`: 希望加载的 iter 数量，可以是`数值`、`latest` 或者 `release`。
- `SAVE_CKPT_DIR`: 转换后的 Hugging Face 模型存储路径。

示例：

```bash
# demo
export MODEL_TYPE="minicpm-1b"
export SEQ_LEN="4k"
export CKPT_PATH="/user/zhangyixuan/models/opencompass_regression/megatron_raw/checkpoints.2542"
export CKPT_ITER="5000"
export SAVE_CKPT_DIR="work_dirs/hf"
bash cybertron_run_scripts/convert_megatron_ckpt_2_hf/direct_path.sh \
    ${MODEL_TYPE} \
    ${SEQ_LEN} \
    ${CKPT_PATH} \
    ${CKPT_ITER} \
    ${SAVE_CKPT_DIR}
```

### Cybertron Job 路径 (Cybertron Job path)

用于转换在 Cybertron 平台训练的 Megatron 模型，需要指定 Cybertron Job 的路径。 Cybertron 模型存储路径为：`/projects/{PROJECT_ID}-{PROJECT_NAME}/checkpoints/checkpoints.{CKPT_JOB_ID}`。保存成 `${SAVE_CKPT_DIR}/jobid_${CKPT_JOB_ID}-iter_${CKPT_ITER}`。

```bash
bash cybertron_run_scripts/convert_megatron_ckpt_2_hf/cybertron_job_path.sh \
    ${MODEL_TYPE} \
    ${SEQ_LEN} \
    ${CKPT_JOB_DIR} \
    ${CKPT_JOB_ID} \
    ${CKPT_ITER} \
    ${SAVE_CKPT_DIR}
```

- `MODEL_TYPE`: 模型类型，可选择 `minicpm-1b` 或 `minicpm3-4b`。
- `SEQ_LEN`: 模型长度，可选择 `4k` 或 `32k`。
- `CKPT_JOB_DIR`: Cybertron 训练的 Megatron 模型保存路径，如 `/projects/118-hqdata-exp/checkpoints`。
- `CKPT_JOB_ID`: Cybertron Job 的 ID。
- `CKPT_ITER`: 希望加载的 iter 数量，可以是`数值`、`latest` 或者 `release`。
- `SAVE_CKPT_DIR`: 转换后的 Hugging Face 模型存储路径。

示例：

```bash
# e.g.
export MODEL_TYPE="minicpm-1b"
export SEQ_LEN="4k"
export CKPT_JOB_DIR="/projects/118-hqdata-exp/checkpoints"
export CKPT_JOB_ID="2542"
export CKPT_ITER="5000"
export SAVE_CKPT_DIR="work_dirs/hf"
bash cybertron_run_scripts/convert_megatron_ckpt_2_hf/cybertron_job_path.sh \
    ${MODEL_TYPE} \
    ${SEQ_LEN} \
    ${CKPT_JOB_DIR} \
    ${CKPT_JOB_ID} \
    ${CKPT_ITER} \
    ${SAVE_CKPT_DIR}
```

### 检查转换后的权重是否正确

```bash
python cybertron_run_scripts/convert_megatron_ckpt_2_hf/diff_bin_file.py ${MODEL_PATH_1} ${MODEL_PATH_2}
```

检查两个 bin 文件是否相同，用于检验当前脚本和之前转换的模型是否一致。
