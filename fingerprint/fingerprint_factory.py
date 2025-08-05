from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.llmmap.llmmap import LLMmapFingerprint
from fingerprint.huref.huref import HuRefFingerprint
from fingerprint.reef.reef import REEFFingerprint
from fingerprint.pdf.pdf import PDFFingerprint
from fingerprint.gradient.gradient import GradientFingerprint
from fingerprint.trap.trap import TRAPFingerprint
from fingerprint.met.model_equality_testing import METFingerprint


def create_fingerprint_method(config=None, accelerator=None) -> LLMFingerprintInterface:
    method_name = config.get("fingerprint_method", None)
    if method_name == "huref":
        return HuRefFingerprint(config=config, accelerator=accelerator)
    elif method_name == "pdf":
        return PDFFingerprint(config=config, accelerator=accelerator)
    elif method_name == "reef":
        return REEFFingerprint(config=config, accelerator=accelerator)
    elif method_name == "gradient":
        return GradientFingerprint(config=config, accelerator=accelerator)
    elif method_name == "llmmap":
        return LLMmapFingerprint(config=config, accelerator=accelerator)
    elif method_name == "trap":
        return TRAPFingerprint(config=config, accelerator=accelerator)
    elif method_name == "met":
        return METFingerprint(config=config, accelerator=accelerator)
    else:
        raise ValueError(f"Unknown fingerprinting method: {method_name}")
