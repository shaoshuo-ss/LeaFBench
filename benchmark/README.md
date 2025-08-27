# How to Extend New Models

This directory is for implementing and extending various models. You can expand models in several ways:

## 1. Extend Parameter-Altered (Derivative) LLMs
- Add new model configurations directly in the `config/benchmark_config.yaml` file.
- Follow the existing format to specify model name, parameters, paths, etc.
- No extra code is required for this type.

## 2. Extend Parameter-Independent Techniques
- Create a new class that implements the interface defined in `benchmark/model_interface.py`.
- At minimum, implement the `generate` method. See `model_interface.py` for the full interface definition.
- Add your new class to the model pool or relevant logic as needed.

### Example
```python
from benchmark.model_interface import ModelInterface

class MyNewModel(ModelInterface):
    def generate(self, prompt):
        # Implement your generation logic here
        pass
```

## 3. Reference Files
- `config/benchmark_config.yaml`: Model configuration file
- `benchmark/model_interface.py`: Model interface definition

If you have any questions, please refer to the above files or contact the maintainer.
