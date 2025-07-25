HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --multi_gpu main.py \
    --device 5,6,7 \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/llmmap.yaml' \
    --log_path 'logs/'