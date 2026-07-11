import pytest
import json
import logging
from backend.app.core.logging import JSONFormatter
from backend.app.db.outbox import init_outbox, enqueue_outbox_event, get_pending_events, update_event_status
from backend.app.db.session import get_connection
from q2_hybrid_router.router.classifier import HybridRouter

def test_platt_scaling():
    """Verify that temperature-calibrated probabilities sum to 1.0."""
    import os
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    router = HybridRouter(config)
    res = router.route("Normal typical test query about weather")
    
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3
    # Check that calibrated probabilities sum to 1.0
    assert abs(sum(res["probabilities"]) - 1.0) < 0.01

def test_transactional_outbox():
    """Verify outbox lifecycle: enqueuing, retrieving, and updating statuses."""
    import uuid
    init_outbox()
    
    run_id = f"test_run_{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    try:
        # Transactional outbox insert
        enqueue_outbox_event(conn, run_id, "compression_run", {
            "model_name": "mobilenet_v2",
            "ood_calibration": False
        })
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    pending = get_pending_events()
    assert any(p["id"] == run_id for p in pending)
    
    # Mark as completed
    update_event_status(run_id, "completed")
    pending_after = get_pending_events()
    assert not any(p["id"] == run_id for p in pending_after)

def test_anomaly_detection():
    """Verify that anomalous inputs are rejected with cloud fallback and flag set."""
    import os
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    router = HybridRouter(config)
    
    # Empty query check
    res_empty = router.route("   ")
    assert res_empty["decision"] == "cloud"
    assert res_empty.get("anomaly_flag") is True
    assert "Empty" in res_empty.get("anomaly_reason")

    # Long query anomaly check (>1000 words)
    long_query = " ".join(["gibberish"] * 1005)
    res_long = router.route(long_query)
    assert res_long["decision"] == "cloud"
    assert res_long.get("anomaly_flag") is True
    assert "length" in res_long.get("anomaly_reason")

    # Character distribution gibberish anomaly check
    gibberish_query = "@#$@#$!@#$!@#$!@#$!@#$!@#$!@#$"
    res_gibberish = router.route(gibberish_query)
    assert res_gibberish["decision"] == "cloud"
    assert res_gibberish.get("anomaly_flag") is True
    assert "Gibberish" in res_gibberish.get("anomaly_reason")

def test_structured_json_logging():
    """Verify that JSONFormatter structures standard logs correctly."""
    formatter = JSONFormatter()
    log_record = logging.LogRecord(
        name="test-logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=10,
        msg="Structured logging test message",
        args=None,
        exc_info=None
    )
    formatted = formatter.format(log_record)
    parsed = json.loads(formatted)
    
    assert "timestamp" in parsed
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Structured logging test message"
    assert parsed["logger"] == "test-logger"

def test_queue_provider_abstraction():
    """Verify that SQLiteOutboxProvider implements BaseQueueProvider and behaves correctly."""
    import uuid
    from backend.app.core.state import queue_provider
    from backend.app.core.queue_protocol import BaseQueueProvider
    from backend.app.db.session import get_connection
    
    assert isinstance(queue_provider, BaseQueueProvider)
    
    provider_run_id = f"test_provider_{uuid.uuid4().hex[:8]}"
    # Verify enqueue behavior
    conn = get_connection()
    try:
        queue_provider.enqueue(
            event_id=provider_run_id,
            event_type="compression_run",
            payload={"model_name": "phi_3_mini", "ood_calibration": True},
            connection=conn
        )
        conn.commit()
    finally:
        conn.close()
        
    from backend.app.db.outbox import get_pending_events, update_event_status
    pending = get_pending_events()
    assert any(p["id"] == provider_run_id for p in pending)
    
    # Mark as completed
    update_event_status(provider_run_id, "completed")

def test_dynamic_retraining_trigger():
    """Verify that dynamic retraining is triggered upon high PSI drift."""
    import os
    import yaml
    from collections import deque
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "q2_hybrid_router", "config", "router_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    router = HybridRouter(config)
    assert router.is_trained
    
    # Set history to trigger a high-drift alert
    router.input_feature_history = deque(maxlen=router.rolling_window)
    # Inject OOD long queries to force drift
    extreme_query = " ".join(["word"] * 450)
    for _ in range(100):
         router.route(extreme_query)
         
    # Compute drift should alert and trigger dynamic retraining
    psi, status = router.compute_drift_psi()
    assert status == "alert" or len(router.input_feature_history) == 0
    # Because compute_drift_psi() triggers dynamic_retrain(), history should be cleared
    assert len(router.input_feature_history) == 0

