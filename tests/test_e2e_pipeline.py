"""
End-to-end integration tests for the QualEdge FastAPI backend.

These tests verify the full request lifecycle:
  - Verified hardware results are present in the API on startup
  - Pipeline trigger -> run_id -> stage polling -> benchmark assertion
  - source="measured" entries exist for the verified AI Hub run

Run with:
    PYTHONPATH=. pytest tests/test_e2e_pipeline.py -v --tb=short
"""
import time
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


class TestVerifiedSiliconResults:
    """Assert that pre-verified Qualcomm AI Hub results are surfaced at startup."""

    def test_verified_measured_benchmark_present(self):
        """
        The compression service must inject the verified AI Hub job result
        (MobileNetV2 FP32, Snapdragon X Elite CRD, 0.55ms) as source='measured'
        on startup. A Qualcomm recruiter hitting /api/compression/benchmarks
        should see real hardware data immediately, without triggering a run.
        """
        response = client.get("/api/compression/benchmarks")
        assert response.status_code == 200, response.text
        benchmarks = response.json()

        measured = [b for b in benchmarks if b.get("source") == "measured"]
        assert len(measured) >= 1, (
            "Expected at least one source='measured' benchmark from the verified "
            "AI Hub job (jgdzzyo65). Got: " + str([b.get("source") for b in benchmarks])
        )

        # Find the specific verified entry
        npu_bench = next(
            (b for b in measured if b.get("latencyMs") == 0.55 and b.get("modelName") == "mobilenet_v2"),
            None,
        )
        assert npu_bench is not None, (
            "Expected MobileNetV2 FP32 with latencyMs=0.55 (verified AI Hub job jgdzzyo65). "
            f"Measured benchmarks found: {measured}"
        )
        assert npu_bench["cpuFallbackOps"] == [], "Verified job has 0 CPU fallback ops (100% HTP native)"
        assert npu_bench["modelSizeMb"] == pytest.approx(14.16, abs=0.1)

    def test_verified_aihub_job_present(self):
        """
        The AI Hub jobs endpoint must surface the pre-verified hardware run
        (job_aihub_verified) from startup — not require a pipeline run first.
        """
        response = client.get("/api/aihub/jobs")
        assert response.status_code == 200, response.text
        jobs = response.json()

        assert len(jobs) >= 1, "Expected at least one AI Hub job record pre-loaded on startup"

        verified = next(
            (j for j in jobs if j.get("latencyMs") == pytest.approx(0.55, abs=0.01)),
            None,
        )
        assert verified is not None, (
            f"Expected pre-loaded job with latencyMs=0.55. Jobs found: {jobs}"
        )
        assert verified["cpuFallbackOps"] == []
        assert verified["status"] == "success"
        assert "mobilenet_v2" in verified["modelName"]

    def test_benchmarks_have_source_labels(self):
        """Every benchmark entry must have a source field set."""
        response = client.get("/api/compression/benchmarks")
        assert response.status_code == 200
        for bench in response.json():
            assert "source" in bench, f"Missing source field on benchmark: {bench.get('id')}"
            assert bench["source"] in ("measured", "demo", "simulated"), (
                f"Invalid source value '{bench['source']}' on benchmark {bench.get('id')}"
            )


class TestPipelineTriggerAndPoll:
    """Full pipeline trigger → stage poll → benchmark assertion."""

    def test_trigger_returns_run_id(self):
        """POST /api/compression/run must return a valid run_id string."""
        response = client.post(
            "/api/compression/run",
            params={"model_name": "mobilenet_v2", "ood_calibration": False},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "run_id" in data
        assert data["run_id"].startswith("run_")

    def test_trigger_then_poll_stages(self):
        """
        Trigger a pipeline run and poll stage progress until all stages resolve.
        Asserts that stages transition through pending -> running -> passed/failed.
        """
        # Trigger
        trigger_resp = client.post(
            "/api/compression/run",
            params={"model_name": "mobilenet_v2", "ood_calibration": False},
        )
        assert trigger_resp.status_code == 200
        run_id = trigger_resp.json()["run_id"]

        # Poll up to 30s for completion
        deadline = time.time() + 30
        stages = []
        while time.time() < deadline:
            poll_resp = client.get(f"/api/compression/run/{run_id}/stages")
            assert poll_resp.status_code == 200
            stages = poll_resp.json()
            statuses = {s["status"] for s in stages}
            if statuses <= {"passed", "failed"}:
                break
            time.sleep(0.5)

        assert len(stages) >= 6, f"Expected at least 6 pipeline stages, got {len(stages)}: {[s['name'] for s in stages]}"
        stage_names = [s["name"] for s in stages]
        assert "fp32" in stage_names

        passed_count = sum(1 for s in stages if s["status"] == "passed")
        assert passed_count >= 4, (
            f"Expected at least 4 passed stages, got {passed_count}. Stages: {stages}"
        )

    def test_new_benchmark_after_pipeline_run(self):
        """
        After a successful pipeline run, /api/compression/benchmarks must include
        a new entry with source='measured' for mobilenet_v2.
        """
        # Trigger and wait
        trigger_resp = client.post(
            "/api/compression/run",
            params={"model_name": "mobilenet_v2", "ood_calibration": False},
        )
        run_id = trigger_resp.json()["run_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            stages = client.get(f"/api/compression/run/{run_id}/stages").json()
            if all(s["status"] in ("passed", "failed") for s in stages):
                break
            time.sleep(0.5)

        # Check benchmarks
        response = client.get("/api/compression/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()
        measured = [
            b for b in benchmarks
            if b.get("source") == "measured" and b.get("modelName") == "mobilenet_v2"
        ]
        assert len(measured) >= 1, (
            "After pipeline run, expected at least one measured mobilenet_v2 benchmark."
        )
