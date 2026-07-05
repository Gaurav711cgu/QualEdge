import os
import uuid
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("AIHub-Client")

# Try to import Qualcomm AI Hub SDK
QAI_HUB_AVAILABLE = False
try:
    import qai_hub as hub
    QAI_HUB_AVAILABLE = True
    logger.info("Qualcomm AI Hub SDK loaded successfully.")
except ImportError:
    logger.warning("Qualcomm AI Hub SDK not found. Running AI Hub actions in SIMULATION mode.")

class AIHubCoordinator:
    """
    Coordinates model compilation and profiling on the Qualcomm AI Hub.
    Submits models for QNN / ONNX execution on the Snapdragon X Elite CRD target.
    """
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("QAI_HUB_API_TOKEN")
        self.client = None
        self.active_mode = False
        
        if QAI_HUB_AVAILABLE and self.api_token:
            try:
                # Configure client with token via environment variable
                os.environ["QAI_HUB_API_TOKEN"] = self.api_token
                self.client = hub.Client()
                self.active_mode = True
                logger.info("Qualcomm AI Hub Client initialized in ACTIVE mode.")
            except Exception as e:
                logger.error(f"Failed to initialize AI Hub client: {str(e)}. Falling back to simulation.")
        
    def submit_compile(self, model_name: str, model_path: str, runtime: str) -> Dict[str, Any]:
        """
        Submits a compilation job to Snapdragon X Elite CRD.
        """
        job_id = f"compile_{uuid.uuid4().hex[:8]}"
        device_name = "Snapdragon X Elite CRD"
        
        if self.active_mode and self.client:
            try:
                # Convert runtime choice to compile target option
                # Target options: --target_runtime qnn, --target_runtime onnx, etc.
                # Pass the runtime choice directly as target option (e.g. qnn_context_binary)
                target_runtime = runtime
                
                logger.info(f"Submitting native compile job to Qualcomm AI Hub for {model_name}...")
                
                # Setup input specifications
                # (vision, audio, and language specific dummy mappings)
                input_specs = {}
                if "mobilenet" in model_name.lower():
                    input_specs = {"image": (1, 3, 224, 224)}
                elif "whisper" in model_name.lower():
                    input_specs = {"input_features": (1, 80, 3000)}
                else:  # LLM (Phi-3)
                    input_specs = {"input_ids": (1, 512), "attention_mask": (1, 512)}
                
                # Submit job
                job = self.client.submit_compile_job(
                    model=model_path,
                    device=hub.Device(device_name),
                    input_specs=input_specs,
                    options=f"--target_runtime {target_runtime}"
                )
                
                return {
                    "compile_job_id": job.job_id,
                    "status": "running",
                    "device": device_name,
                    "runtime": runtime,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Native AI Hub compile submission failed: {str(e)}")
                return {
                    "compile_job_id": job_id,
                    "status": "failed",
                    "device": device_name,
                    "runtime": runtime,
                    "error": str(e)
                }
        
        # --- SIMULATED COMPILE JOB ---
        logger.info(f"Simulating AI Hub compilation for {model_name} targeting {runtime}...")
        return {
            "compile_job_id": job_id,
            "status": "success",
            "device": device_name,
            "runtime": runtime,
            "error": None
        }

    def submit_profile(self, compile_job_id: str, model_name: str, runtime: str) -> Dict[str, Any]:
        """
        Submits a profiling job to measure latency on the Snapdragon X Elite CRD target.
        """
        profile_job_id = f"profile_{uuid.uuid4().hex[:8]}"
        device_name = "Snapdragon X Elite CRD"
        
        if self.active_mode and self.client:
            try:
                logger.info(f"Submitting native profile job for compile job {compile_job_id}...")
                compile_job = self.client.get_job(compile_job_id)
                compiled_model = compile_job.get_target_model()
                
                job = self.client.submit_profile_job(
                    model=compiled_model,
                    device=hub.Device(device_name)
                )
                
                return {
                    "profile_job_id": job.job_id,
                    "status": "running",
                    "error": None
                }
            except Exception as e:
                logger.error(f"Native AI Hub profile submission failed: {str(e)}")
                return {
                    "profile_job_id": profile_job_id,
                    "status": "failed",
                    "error": str(e)
                }

        # --- SIMULATED PROFILE JOB ---
        # Generate realistic Qualcomm hardware latencies and CPU fallback indicators
        cpu_fallbacks = []
        if "mobilenet" in model_name.lower():
            if "fp32" in runtime:
                latency = 12.50
                cpu_fallbacks = ["Resize", "Softmax"] # FP32 has high fallbacks on basic runtimes
            elif "w8a8" in runtime:
                latency = 1.25  # Fast NPU path
            else:
                latency = 0.95  # Fast INT4 path
        elif "whisper" in model_name.lower():
            if "fp32" in runtime:
                latency = 85.0
                cpu_fallbacks = ["LayerNormalization", "MultiHeadAttention"]
            elif "w8a8" in runtime:
                latency = 18.5
                cpu_fallbacks = ["LayerNormalization"]  # HTP lack of float LN support
            else:
                latency = 14.2
                cpu_fallbacks = ["LayerNormalization"]
        else: # Phi-3 LLM
            if "fp32" in runtime:
                latency = 450.0
                cpu_fallbacks = ["Embedding", "LayerNorm", "RotaryEmbedding"]
            elif "w8a8" in runtime:
                latency = 55.0
                cpu_fallbacks = ["LayerNorm"]
            else:
                latency = 28.0  # INT4 token throughput optimization
                cpu_fallbacks = ["LayerNorm"]

        return {
            "profile_job_id": profile_job_id,
            "status": "success",
            "latency_ms": latency,
            "cpu_fallback_ops": cpu_fallbacks,
            "error": None
        }

    def poll_job(self, job_type: str, job_id: str) -> Dict[str, Any]:
        """
        Polls the status of compile or profile jobs.
        """
        if self.active_mode and self.client:
            try:
                if job_type == "compile":
                    job = self.client.get_job(job_id)
                else:
                    job = self.client.get_job(job_id)
                
                status = job.get_status()
                
                # Map status codes
                status_map = {
                    "SUCCESS": "success",
                    "FAILED": "failed",
                    "IN_PROGRESS": "running",
                    "QUEUED": "queued"
                }
                mapped_status = status_map.get(status.code, "running")
                logger.info(f"AI Hub job {job_id} raw status: {status.code} -> mapped to: {mapped_status}")
                
                latency_ms = None
                cpu_fallbacks = []
                
                if job_type == "profile" and mapped_status == "success":
                    profile_data = job.download_profile()
                    # Extract latency from downloaded profile JSON (in microseconds)
                    summary = profile_data.get("execution_summary", {})
                    latency_us = summary.get("estimated_inference_time", 0)
                    if latency_us == 0 and summary.get("all_inference_times"):
                        import statistics
                        latency_us = statistics.median(summary.get("all_inference_times"))
                    latency_ms = latency_us / 1000.0
                    
                    # Parse out cpu fallbacks by scanning execution layers
                    detail = profile_data.get("execution_detail", [])
                    cpu_fallbacks = list(set([layer.get("type") for layer in detail if layer.get("compute_unit") == "CPU" and layer.get("type")]))
                
                return {
                    "status": mapped_status,
                    "latency_ms": latency_ms,
                    "cpu_fallback_ops": cpu_fallbacks,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Polling native job {job_id} failed: {str(e)}")
                return {
                    "status": "failed",
                    "latency_ms": None,
                    "cpu_fallback_ops": [],
                    "error": str(e)
                }

        # --- SIMULATED POLLING ---
        # Simulated jobs immediately succeed
        return {
            "status": "success",
            "error": None
        }
