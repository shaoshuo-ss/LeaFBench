# LeaFBench

LeaFBench is a comprehensive benchmark designed to evaluate and compare various fingerprinting methods for large language models (LLMs). The project provides a unified interface for running, testing, and analyzing different models and algorithms, facilitating reproducible research and fair comparison.

## Project Structure

- `main.py`: Entry point for running the benchmark.
- `benchmark/`: Core benchmark logic, model interfaces, and implementations.
- `config/`: Configuration files for different experiments and benchmarks.
- `data/`: Datasets and auxiliary files used in experiments.
- `fingerprint/`: Fingerprinting methods and related utilities.
- `scripts/`: Shell scripts for running experiments and benchmarks.
- `utils/`: Utility functions and helpers.

## Environment Setup

**Install dependencies:**
Ensure you have Python 3.8+ installed. Install required packages:
```bash
pip install -r requirements.txt
```

## Reproducing Experiments

- All configuration files required to reproduce the experiments are located in the `config/` directory. Each YAML file corresponds to a specific experiment or benchmark setting.
- All scripts for running experiments are provided in the `scripts/` directory. Simply execute the corresponding shell script to run an experiment. For example:
  ```bash
  bash scripts/gradient.sh
  ```
  The results will be generated according to the configuration specified in the related config file.

For further details, please refer to the comments in the scripts and configuration files.
