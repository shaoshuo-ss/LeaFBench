HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=5,6,7 python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/llmmap.yaml' \
    --log_path 'logs/'