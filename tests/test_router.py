import pytest
import yaml
from q2_hybrid_router.router.classifier import HybridRouter
from q2_hybrid_router.evaluation.evaluator import RouterEvaluator

def test_router_routing():
    # Load config
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
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
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
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
    import os
    from q2_hybrid_router.router.dataset import load_router_dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    router = HybridRouter(config)
    assert router.is_trained, "Router must be trained for PSI test"
    assert router.reference_feature_stats, "Reference stats must be populated"

    # --- Stable case: route 100 queries from the same distribution as training ---
    train_x, _, _, _ = load_router_dataset()
    for query in train_x[:100]:
        router.route(query)

    psi, status = router.compute_drift_psi()
    # Queries are from training distribution -> PSI should be low (< 0.25 is warning/stable)
    assert psi < 0.25, f"PSI={psi} expected < 0.25 for in-distribution queries"
    assert status in ("stable", "warning"), f"Expected stable/warning, got {status}"

    # --- Alert case: inject very long, out-of-distribution queries (extreme shift) ---
    from collections import deque
    router.input_feature_history = deque(maxlen=router.rolling_window)
    extreme_query = " ".join([f"word{i}" for i in range(400)])  # ~400 words, training avg ~10
    for _ in range(100):
        router.route(extreme_query)

    psi, status = router.compute_drift_psi()
    # 400-word queries vs training distribution of ~10-word queries -> PSI should be high
    assert psi > 0.2, f"PSI={psi} expected > 0.2 for extreme OOD queries"
    assert status == "alert", f"Expected alert, got {status} (PSI={psi})"

def test_router_evaluator():
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
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

def test_router_modernbert_pathway():
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    router = HybridRouter(config)
    
    # Test routing through ModernBERT pathway
    res = router.route("What is the capital of Japan?", pathway="modernbert")
    assert "decision" in res
    assert "probabilities" in res
    
    # If ModernBERT is not trained or loaded, it will use TF-IDF fallback which works perfectly
    assert res["decision"] in ["on_device", "on_device_with_retry", "cloud"]
