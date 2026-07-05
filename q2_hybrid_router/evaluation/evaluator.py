import numpy as np
from typing import Dict, Any, List, Tuple
from q2_hybrid_router.router.classifier import HybridRouter
from q2_hybrid_router.router.dataset import load_router_dataset

class RouterEvaluator:
    def __init__(self, router: HybridRouter):
        self.router = router
        self.train_x, self.train_y, self.test_x, self.test_y = load_router_dataset()

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs the router across the test dataset and compiles performance statistics:
        - Accuracy of routing decision
        - Average end-to-end latency
        - Average cost per 1000 queries
        - Average quality score (0-100 scale)
        """
        correct = 0
        total = len(self.test_x)
        
        latencies = []
        costs = []
        quality_scores = []
        routing_decisions = []

        for query, ground_truth in zip(self.test_x, self.test_y):
            route_result = self.router.route(query)
            decision = route_result["decision"]
            
            # Map decision to numeric code
            dec_code = {"on_device": 0, "on_device_with_retry": 1, "cloud": 2}[decision]
            routing_decisions.append(dec_code)
            
            # We measure classification correctness. Note that routing doesn't have to match 
            # ground truth exactly (it can be conservative), but matching indicates high routing accuracy.
            if dec_code == ground_truth:
                correct += 1
                
            # Compute estimated end-to-end latency & cost for the path
            if decision == "on_device":
                # Router latency + local inference latency
                latencies.append(route_result["router_latency_ms"] + 80.0)
                costs.append(0.0)
                quality_scores.append(72.0)  # On-device model baseline quality
            elif decision == "on_device_with_retry":
                # Simulated retry logic: 80% success locally, 20% fallback to cloud
                local_succ = np.random.rand() > 0.20
                if local_succ:
                    latencies.append(route_result["router_latency_ms"] + 80.0)
                    costs.append(0.0)
                    quality_scores.append(85.0)  # Self-verification checks boost quality
                else:
                    # Local inference (80ms) + verification check (5ms) + cloud inference (800ms)
                    latencies.append(route_result["router_latency_ms"] + 80.0 + 5.0 + 800.0)
                    costs.append(0.0055)  # Cost of cloud fallback
                    quality_scores.append(92.0)
            else:  # Direct cloud
                latencies.append(route_result["router_latency_ms"] + 800.0)
                costs.append(0.0055)
                quality_scores.append(95.0)

        # Summarize results
        avg_latency = float(np.mean(latencies))
        avg_cost_1k = float(np.mean(costs)) * 1000.0
        avg_quality = float(np.mean(quality_scores))
        routing_accuracy = (correct / total) * 100.0
        
        # Calculate rates
        dec_array = np.array(routing_decisions)
        on_dev_rate = float(np.sum(dec_array == 0) / total) * 100.0
        retry_rate = float(np.sum(dec_array == 1) / total) * 100.0
        cloud_rate = float(np.sum(dec_array == 2) / total) * 100.0

        return {
            "routing_accuracy": round(routing_accuracy, 2),
            "average_latency_ms": round(avg_latency, 2),
            "average_cost_per_1k": round(avg_cost_1k, 4),
            "average_quality_score": round(avg_quality, 2),
            "on_device_rate": round(on_dev_rate, 2),
            "on_device_with_retry_rate": round(retry_rate, 2),
            "cloud_rate": round(cloud_rate, 2),
            "total_queries": total
        }

    def sweep_thresholds(self) -> List[Dict[str, Any]]:
        """
        Sweeps the complexity threshold slider from 0.05 to 0.95, mapping the tradeoff
        frontier for the Pareto chart on the frontend.

        FIX: Uses copy.deepcopy() to create an isolated router instance for the sweep.
        Previously, this mutated self.router.thresholds in-place while the live API
        may be routing real requests with the modified threshold, and each call to
        self.router.route() was appending to input_feature_history, contaminating PSI
        data with sweep traffic. The deep copy prevents both issues.
        """
        import copy
        sweep_points = []

        for thresh in np.linspace(0.05, 0.95, 10):
            # Isolated copy — the live router is never touched
            sweep_router = copy.deepcopy(self.router)
            sweep_router.thresholds["on_device_limit"] = float(thresh)

            # Run benchmark on the isolated copy
            evaluator_copy = RouterEvaluator(sweep_router)
            res = evaluator_copy.run_benchmark()

            false_neg_count = 0
            complex_query_count = 0
            for query, ground_truth in zip(self.test_x, self.test_y):
                if ground_truth == 2:
                    complex_query_count += 1
                    route_result = sweep_router.route(query)
                    if route_result["decision"] == "on_device":
                        false_neg_count += 1

            fn_rate = (false_neg_count / max(complex_query_count, 1)) * 100.0

            sweep_points.append({
                "threshold": round(float(thresh), 2),
                "falseNegativeRate": round(fn_rate, 2),
                "cloudRate": res["cloud_rate"],
                "onDeviceRate": res["on_device_rate"] + res["on_device_with_retry_rate"],
                "estimatedCostPerThousand": res["average_cost_per_1k"],
                "p95LatencyMs": round(res["average_latency_ms"] * 1.25, 2)
            })

        # self.router is completely untouched — live routing and PSI are unaffected
        return sweep_points
