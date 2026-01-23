HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/mdir.yaml' \
    --log_path 'logs/'