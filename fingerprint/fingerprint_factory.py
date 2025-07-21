from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.llmmap.llmmap import LLMmap
from fingerprint.huref.huref import HuRef

def create_fingerprint_method(config=None, accelerator=None) -> LLMFingerprintInterface:
    method_name = config.get("fingerprint_method", None)
    if method_name == "llmmap":
        return LLMmap(config=config, accelerator=accelerator)
    elif method_name == "huref":
        return HuRef(config=config, accelerator=accelerator)
    else:
        raise ValueError(f"Unknown fingerprinting method: {method_name}")
