import pytest
import yaml
from q1_compression_suite.compression.pipeline import AIMETCompressionPipeline

def test_pipeline_simulation():
    # Load config
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q1_compression_suite/config/compression_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = AIMETCompressionPipeline("mobilenet_v2", config)
    results = pipeline.run_pipeline()
    
    # Assert stages exist and run successfully
    assert len(results) == 6
    stages = [r["stage"] for r in results]
    assert "fp32" in stages
    assert "bn_fold" in stages
    assert "cle" in stages
    assert "relu6_replace" in stages
    assert "adaround_w8a8" in stages
    assert "adaround_w4a8" in stages
    
    # Assert size shrinks
    assert results[-1]["model_size_mb"] < results[0]["model_size_mb"]

def test_pipeline_ood_collapse():
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q1_compression_suite/config/compression_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Test Out-Of-Distribution calibration collapse
    pipeline = AIMETCompressionPipeline("mobilenet_v2", config, ood_calibration=True)
    results = pipeline.run_pipeline()
    
    # MobileNetV2 FP32 vs W4A8 OOD
    fp32_acc = results[0]["metric_value"]
    w4a8_acc = results[-1]["metric_value"]
    
    # Massive collapse under OOD
    assert fp32_acc > 70.0
    assert w4a8_acc < 10.0
