# How to Extend New Fingerprinting Methods

This directory is for implementing and extending various model fingerprinting methods.

## 1. Implement a New Fingerprinting Method
- Create a new class following `fingerprint/fingerprint_interface.py`, and implement the three required methods (such as `fit`, `extract`, `compare`, etc.—refer to the interface for details).
- Ensure your implementation is compatible with the interface.

### Example
```python
from fingerprint.fingerprint_interface import FingerprintInterface

class MyNewFingerprint(FingerprintInterface):
    def fit(self, data):
        # Training/initialization logic
        pass
    def extract(self, model):
        # Fingerprint extraction logic
        pass
    def compare(self, fp1, fp2):
        # Fingerprint comparison logic
        pass
```

## 2. Register the New Method
- Register your new fingerprinting method in `fingerprint/fingerprint_factory.py`.
- Follow the existing registration pattern to add your class to the factory or registry.

## 3. Reference Files
- `fingerprint/fingerprint_interface.py`: Fingerprinting method interface definition
- `fingerprint/fingerprint_factory.py`: Fingerprinting method registration and factory

If you have any questions, please refer to the above files or contact the maintainer.
