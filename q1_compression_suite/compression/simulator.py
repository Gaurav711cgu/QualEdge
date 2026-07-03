import time
import random
from typing import Dict, Any, List

class CompressionSimulator:
    """
    Simulates the AIMET post-training quantization pipeline stages,
    generating realistic output metrics, file paths, and calibration sensitivity
    behavior for vision, audio, and language models.
    """
    
    # Baseline stats: (acc/wer/ppl, size_mb, base_latency_ms)
    MODEL_BASELINES = {
        "mobilenet_v2": {
            "family": "vision",
            "metric_name": "top1_accuracy",
            "fp32_val": 71.88,
            "size_mb": 14.3,
            "cpu_latency": 12.5,
            "htp_latency": 1.2
        },
        "whisper_tiny": {
            "family": "audio",
            "metric_name": "wer",
            "fp32_val": 12.15,
            "size_mb": 151.0,
            "cpu_latency": 85.0,
            "htp_latency": 8.5
        },
        "phi_3_mini": {
            "family": "language",
            "metric_name": "perplexity",
            "fp32_val": 10.45,
            "size_mb": 7600.0,  # 3.8B model in FP16/FP32 size
            "cpu_latency": 450.0,
            "htp_latency": 28.0
        }
    }

    def simulate_stage(self, model_name: str, stage_name: str, ood_calibration: bool = False) -> Dict[str, Any]:
        """
        Simulates one of the pipeline stages:
        fp32 -> bn_fold -> cle -> relu6_replace -> adaround_w8a8 -> adaround_w4a8
        """
        if model_name not in self.MODEL_BASELINES:
            raise ValueError(f"Unknown model name: {model_name}")

        baseline = self.MODEL_BASELINES[model_name]
        family = baseline["family"]
        metric_name = baseline["metric_name"]
        
        # Calculate metric and size adjustments per stage
        val = baseline["fp32_val"]
        size = baseline["size_mb"]
        
        # Simulate slight time delay for realism
        time.sleep(0.1)

        if stage_name == "fp32":
            notes = "Loaded FP32 PyTorch baseline model."
            
        elif stage_name == "bn_fold":
            # BN folding merges batch norm params into conv weights. Size slightly drops (negligible), acc is identical.
            if family == "vision":
                size *= 0.99
                notes = "Folded batch normalization layers into preceding Conv2d layers. Removed 18 redundant BN layers."
            else:
                notes = "Batch Normalization folding not applicable (no BN layers in model architecture)."
                
        elif stage_name == "cle":
            # CLE scales activations across layers to narrow weight distributions. Accuracy stays steady.
            if family in ["vision", "audio"]:
                notes = "Applied Cross-Layer Equalization. Balanced weight ranges across 12 layer pairs."
            else:
                notes = "Cross-Layer Equalization completed. Verified range balancing on Self-Attention layers."
                
        elif stage_name == "relu6_replace":
            # Swaps ReLU6 (which clamps range) for standard ReLU to avoid clipping in subsequent quantization scaling.
            notes = "Replaced ReLU6 activations with standard ReLU for better dynamic quantization range."
            
        elif stage_name == "adaround_w8a8":
            if ood_calibration:
                # OOD calibration results in bad quantization parameter selection
                if family == "vision":
                    val = 22.4  # Masssive accuracy drop
                elif family == "audio":
                    val = 88.5  # High Word Error Rate
                else:
                    val = 1450.2 # Exploded Perplexity
                notes = "WARNING: Calibration dataset is Out-Of-Distribution! Quantization reconstruction error minimized on noise; scale parameters clipping valid activations."
            else:
                # Standard INT8 AdaRound: minimal degradation
                if family == "vision":
                    val = val - random.uniform(0.15, 0.35)
                elif family == "audio":
                    val = val + random.uniform(0.2, 0.5)
                else:
                    val = val + random.uniform(0.3, 0.7)
                notes = "AdaRound INT8 weight rounding complete. Reconstruction error minimized."
            size *= 0.25  # INT8 is 1/4 the size of FP32
            
        elif stage_name == "adaround_w4a8":
            if ood_calibration:
                if family == "vision":
                    val = 4.2  # Virtual collapse
                elif family == "audio":
                    val = 99.8
                else:
                    val = 15000.0
                notes = "CRITICAL: W4A8 AdaRound collapsed under Out-Of-Distribution calibration. Outliers clipped completely."
            else:
                # Standard W4A8 AdaRound: moderate degradation, but high size savings
                if family == "vision":
                    val = val - random.uniform(2.5, 4.5)
                elif family == "audio":
                    val = val + random.uniform(4.0, 7.5)  # Audio features are sensitive
                else:
                    val = val + random.uniform(2.5, 5.0)  # Language embeds are sensitive
                notes = "AdaRound INT4/INT8 weight-activation optimization complete. Sensitivity outliers mitigated."
            size *= 0.14  # W4A8 is ~1/7th size of FP32 (4-bit weights, 8-bit activations)
            
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        return {
            "model_name": model_name,
            "stage": stage_name,
            "metric_name": metric_name,
            "metric_value": round(val, 2),
            "model_size_mb": round(size, 2),
            "notes": notes,
            "status": "passed"
        }
