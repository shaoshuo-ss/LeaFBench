from utils.utils import load_args, setup_environment, init_log, load_config
from accelerate import Accelerator
import logging
from benchmark.benchmark import Benchmark
from fingerprint.fingerprint_factory import create_fingerprint_method

if __name__ == "__main__":
    args = load_args()
    
    # Initialize logging
    logger = init_log(args)

    # Setup environment
    accelerator, device = setup_environment(args)
    
    # Load configurations
    benchmark_config = load_config(args.benchmark_config)
    fingerprint_config = load_config(args.fingerprint_config)
    
    logger.info("Configuration loaded successfully!")
    logger.info(f"Benchmark config: {benchmark_config}")
    logger.info(f"Fingerprint config: {fingerprint_config}")
    
    # initialize the benchmark with accelerator
    benchmark = Benchmark(benchmark_config, accelerator=accelerator)

    # initialize the fingerprint method with accelerator
    fingerprint_method = create_fingerprint_method(fingerprint_config, accelerator=accelerator)

    logger.info(f"Fingerprint method {fingerprint_config.get('fingerprint_method')} initialized with accelerator support")
    logger.info(f"Benchmark initialized with {len(benchmark.list_available_models())} models")
    logger.info(f"Available models: {benchmark.list_available_models()}")

    # prepare fingerprint method (this may involve training)
    logger.info("Preparing fingerprint method...")
    fingerprint_method.prepare(train_models=benchmark.get_training_models())

    # evaluate fingerprinting method
    
