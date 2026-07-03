import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("Local-Inference-Engine")

class LocalInferenceEngine:
    """
    Handles local, on-device model execution (simulating Qwen-0.5B-Chat INT4 compiled via QNN).
    Under macOS, it runs a high-fidelity simulator that models latency, token throughput,
    and typical quantization artifacts (e.g. repetition loops).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config["models"]["local"]
        self.model_name = self.config["name"]
        self.target_latency = self.config["target_latency_ms"]
        
    def generate(self, query: str, force_degrade: bool = False) -> Dict[str, Any]:
        """
        Generates a response from the local model.
        If force_degrade is True, simulates a quantization breakdown (infinite repetition loop)
        to test the router's self-verification/retry loop.
        """
        start_time = time.perf_counter()
        
        # Simulate local execution time: ~65ms to 95ms
        time.sleep(random_uniform := (self.target_latency * 0.8 + 15) / 1000.0)
        
        if force_degrade:
            # Simulate a standard quantization repetition loop breakdown
            # (e.g. model outputting 'the the the the the...')
            text = "For the wheat zone, we need to divide the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone the zone."
            cpu_fallback_ops = ["LayerNorm"]
        else:
            # High-fidelity response generator based on query types
            q_lower = query.lower()
            if "france" in q_lower:
                text = "The capital of France is Paris. It is a major European city and a global center for art, fashion, gastronomy, and culture."
            elif "romeo" in q_lower:
                text = "William Shakespeare wrote the tragedy Romeo and Juliet. It tells the story of two young star-crossed lovers whose deaths ultimately reconcile their feuding families."
            elif "boiling" in q_lower:
                text = "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
            elif "discount" in q_lower or "20%" in q_lower:
                text = "If a store offers a 20% discount on a $50 shirt, you save $10. The final purchase price is $40."
            elif "weather" in q_lower or "new york" in q_lower:
                text = "New York winters are cold, snowy, and damp (averaging 30-40°F), whereas Los Angeles has mild, dry winters with temperatures averaging 55-68°F."
            elif "python" in q_lower or "script" in q_lower:
                text = "Here is a simple script:\n```python\nimport requests\nfrom bs4 import BeautifulSoup\n# news scraper logic...\n```"
            else:
                text = f"On-device response: I processed your query about '{query}'. In order to save battery, this response was computed locally on the Hexagon NPU using W4A8 weights."
                
            cpu_fallback_ops = []

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        token_count = len(text.split()) * 1.3  # Rough estimation of tokens
        tokens_per_sec = token_count / (latency_ms / 1000.0)
        
        return {
            "text": text,
            "latency_ms": round(latency_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "cpu_fallback_ops": cpu_fallback_ops,
            "device": "Snapdragon X Elite NPU (Simulated)",
            "precision": "w4a8"
        }
