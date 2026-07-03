import pytest
import yaml
from q2_hybrid_router.router.classifier import HybridRouter
from q2_hybrid_router.evaluation.evaluator import RouterEvaluator

def test_router_routing():
    # Load config
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q2_hybrid_router/config/router_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    router = HybridRouter(config)
    
    # Test simple query routing
    res_simple = router.route("What is the capital of Japan?")
    assert res_simple["decision"] == "on_device"
    
    # Test complex query routing
    res_complex = router.route("Write a Python script that scrapes headlines from a news website and sends an email digest.")
    assert res_complex["decision"] == "cloud"

def test_router_verification_retry():
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q2_hybrid_router/config/router_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    router = HybridRouter(config)
    
    # Test normal response
    assert router.verify_local_output("query", "The capital of Japan is Tokyo.") is True
    
    # Test empty response
    assert router.verify_local_output("query", "") is False
    
    # Test repetitive local model collapse
    assert router.verify_local_output("query", "tokyo tokyo tokyo tokyo tokyo tokyo tokyo tokyo tokyo tokyo tokyo tokyo") is False

def test_router_drift_psi():
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q2_hybrid_router/config/router_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    router = HybridRouter(config)
    
    # Injecting stable distribution matching training set exactly
    router.rolling_decisions = []
    for i, pct in enumerate(router.train_distribution):
        count = int(round(pct * 100))
        router.rolling_decisions.extend([i] * count)
        
    psi, status = router.compute_drift_psi()
    assert psi < 0.1
    assert status == "stable"
    
    # Injecting shifted distribution (e.g., massive spike in complex queries -> cloud)
    router.rolling_decisions = []
    for _ in range(50):
        router.rolling_decisions.append(2)  # All complex
        
    psi, status = router.compute_drift_psi()
    assert psi > 0.2
    assert status == "alert"

def test_router_evaluator():
    config_path = "/Users/gauravkumarnayak/Desktop/edgeai-suite/q2_hybrid_router/config/router_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    router = HybridRouter(config)
    evaluator = RouterEvaluator(router)
    
    stats = evaluator.run_benchmark()
    assert "routing_accuracy" in stats
    assert "average_latency_ms" in stats
    assert "average_cost_per_1k" in stats
    
    sweep = evaluator.sweep_thresholds()
    assert len(sweep) == 10
    assert sweep[0]["threshold"] < sweep[-1]["threshold"]
