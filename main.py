from benchmark import base_models
from utils.utils import load_args, setup_environment, init_log, load_config
import pandas as pd
import os
from benchmark.benchmark import Benchmark, save_fingerprints, load_fingerprints
from fingerprint.fingerprint_factory import create_fingerprint_method


if __name__ == "__main__":
    args = load_args()
    
    # Load configurations
    benchmark_config = load_config(args.benchmark_config)
    fingerprint_config = load_config(args.fingerprint_config)
    args.seed = fingerprint_config.get("seed", 42)

    args.fingerprint_method = fingerprint_config.get("fingerprint_method", None)
    if args.fingerprint_method is None:
        raise ValueError("Fingerprint method must be specified in the fingerprint configuration file.")

    # Initialize logging
    logger = init_log(args)
    
    # save config to args.save_path
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        config_save_path = os.path.join(args.save_path, "config.json")
        with open(config_save_path, 'w') as f:
            pd.DataFrame([benchmark_config, fingerprint_config]).to_json(f, orient='records', indent=4)
        logger.info(f"Configuration saved to {config_save_path}")
    logger.info("Configuration loaded successfully!")
    logger.info(f"Benchmark config: {benchmark_config}")
    logger.info(f"Fingerprint config: {fingerprint_config}")

    # Setup accelerator environment
    accelerator, device = setup_environment(args)
    
    # initialize the benchmark with accelerator
    benchmark = Benchmark(benchmark_config, accelerator=accelerator, 
                          fingerprint_type=fingerprint_config.get("fingerprint_type", "black-box"))

    # initialize the fingerprint method with accelerator
    fingerprint_method = create_fingerprint_method(fingerprint_config, accelerator=accelerator)

    # get cached fingerprints path for potential use in error handling
    cached_fingerprints_path = fingerprint_config.get("cached_fingerprints_path", None)
    
    # resume the cached fingerprints if available (only main process loads, then broadcast)
    if not fingerprint_config.get("re_fingerprinting"):
        # if accelerator.is_main_process:
        load_fingerprints(cached_fingerprints_path, benchmark)
        # Wait for main process to finish loading before continuing
        # accelerator.wait_for_everyone()

    logger.info(f"Fingerprint method {fingerprint_config.get('fingerprint_method')} initialized.")
    logger.info(f"Benchmark initialized with {len(benchmark.list_available_models())} models.")
    logger.info(f"Available models: {benchmark.list_available_models()}")

    # prepare fingerprint method (this may involve training)
    logger.info("Preparing fingerprint method...")
    fingerprint_method.prepare(train_models=benchmark.get_training_models())

    # extracting fingerprints from all the models
    # logger.info("Extracting fingerprints from models...")
    all_models = benchmark.get_all_models()
    try:
        for model_name, model in all_models.items():
            logger.info(f"Extracting fingerprint for model: {model_name}")
            if model.get_fingerprint() is not None:
                logger.info(f"Fingerprint for {model_name} already exists, skipping...")
                continue
            fingerprint = fingerprint_method.get_fingerprint(model)
            model.set_fingerprint(fingerprint)
            logger.info(f"Fingerprint for {model_name} extracted successfully.")
    except Exception as e:
        # save the fingerprints if any error occurs (only on main process)
        if accelerator.is_main_process:
            logger.error(f"Error during fingerprint extraction: {e}")
            if cached_fingerprints_path is not None:
                logger.info("Saving fingerprints to cache due to error...")
                try:
                    save_fingerprints(cached_fingerprints_path, benchmark)
                    logger.info("Fingerprints saved successfully during error handling.")
                except Exception as save_error:
                    logger.error(f"Failed to save fingerprints during error handling: {save_error}")
            else:
                logger.warning("No cached fingerprints path specified, cannot save fingerprints during error.")
        raise
    
    # Save the fingerprints to cache (only on main process)
    if fingerprint_config.get("use_cache", False) and accelerator.is_main_process:
        if cached_fingerprints_path is not None:
            save_fingerprints(cached_fingerprints_path, benchmark)
            logger.info("Fingerprints saved to cache successfully!")
        else:
            logger.warning("No cached fingerprints path specified, cannot save fingerprints to cache.")
    # compare fingerprints of different models using the integrated evaluation method
    if accelerator.is_main_process:
        # logger.info("Comparing fingerprints of models...")
        # results_comparing_pretrained_models = {}
        # results_comparing_instruct_models = {}
        # for model_name, model in all_models.items():

        #     # get the pretrained model for comparison
        #     pretrained_model = benchmark.get_model(model.pretrained_model)
        #     if pretrained_model is None:
        #         logger.warning(f"No pretrained model found for {model_name}, skipping comparison.")
        #         continue
        #     similarity = fingerprint_method.compare_fingerprints(base_model=pretrained_model, testing_model=model)
        #     logger.info(f"Similarity between {model_name} and its pretrained model: {similarity:.4f}")
        #     results_comparing_pretrained_models[model_name] = similarity
            
        #     # get the instruct model for comparison
        #     instruct_model = benchmark.get_model(model.instruct_model)
        #     if instruct_model is None:
        #         logger.warning(f"No instruct model found for {model_name}, skipping comparison.")
        #         continue
        #     instruct_similarity = fingerprint_method.compare_fingerprints(base_model=instruct_model, testing_model=model)
        #     logger.info(f"Similarity between {model_name} and its instruct model: {instruct_similarity:.4f}")
        #     results_comparing_instruct_models[model_name] = instruct_similarity
    
        # # save the results of comparing fingerprints (only on main process)
        # # Create a DataFrame with the comparison results
        # comparison_data = []
        # for model_name in results_comparing_pretrained_models.keys():
        #     pretrained_similarity = results_comparing_pretrained_models.get(model_name, None)
        #     instruct_similarity = results_comparing_instruct_models.get(model_name, None)
        #     comparison_data.append({
        #         'model_name': model_name,
        #         'pretrained_similarity': pretrained_similarity,
        #         'instruct_similarity': instruct_similarity
        #     })
        
        # # Convert to DataFrame and save as CSV
        # df = pd.DataFrame(comparison_data)
        # output_csv_path = os.path.join(args.save_path, f"{fingerprint_config.get('fingerprint_method')}_results.csv")
        # if output_csv_path:
        #     os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        #     df.to_csv(output_csv_path, index=False)
        #     logger.info(f"Comparison results saved to: {output_csv_path}")
        # else:
        #     logger.warning("No results_path specified in fingerprint config. Results not saved.")
        logger.info("Evaluating fingerprinting method...")
        evaluation_results = benchmark.evaluate_fingerprinting_method(
            fingerprint_method=fingerprint_method,
            save_path=args.save_path
        )
        logger.info("Fingerprinting method evaluation completed!")
        
        # Log summary of results
        if 'family_metrics' in evaluation_results:
            logger.info("Model Family Metrics:")
            for family, metrics in evaluation_results['family_metrics'].items():
                logger.info(f"  {family}:")
                for metric_type in ['pretrained_model', 'instruct_model', 'overall']:
                    if metric_type in metrics:
                        auc = metrics[metric_type]['AUC']
                        acc = metrics[metric_type]['Accuracy']
                        samples = metrics[metric_type]['Total_Samples']
                        logger.info(f"    {metric_type}: AUC={auc:.4f}, Accuracy={acc:.4f}, Samples={samples}")
        
        if 'type_metrics' in evaluation_results:
            logger.info("Model Type Metrics:")
            for type_name, metrics in evaluation_results['type_metrics'].items():
                logger.info(f"  {type_name}:")
                for metric_type in ['pretrained_model', 'instruct_model', 'overall']:
                    if metric_type in metrics:
                        auc = metrics[metric_type]['AUC']
                        acc = metrics[metric_type]['Accuracy']
                        samples = metrics[metric_type]['Total_Samples']
                        logger.info(f"    {metric_type}: AUC={auc:.4f}, Accuracy={acc:.4f}, Samples={samples}")
    
    # Wait for all processes to complete before exiting
    accelerator.wait_for_everyone()
    # Properly cleanup the distributed process group
    if accelerator is not None:
        # End the accelerator to properly cleanup distributed resources
        accelerator.end_training()
        
        # Additional cleanup for distributed training
        if hasattr(accelerator.state, 'process_group') and accelerator.state.process_group is not None:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
    
    logger.info("Program completed successfully.")
