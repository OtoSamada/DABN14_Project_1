from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelMetadata:
    """
    Simple experiment tracker for ML models.
    
    Tracks:
      - experiment_name, timestamp, algorithm
      - hyperparameters, results (metrics)
      - data_info, random_state
      - training_time_seconds, inference_time_ms (per sample)
    
    Saves to: models/experiments/<experiment_name>.json
    """

    def __init__(self, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name
        self.metadata_log: List[Dict[str, Any]] = []

    def log_experiment(
        self,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        results: Dict[str, Any],
        data_info: Dict[str, Any],
        random_state: Optional[int] = None,
        training_time_seconds: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
    ) -> None:
        """Log a single experiment run."""
        record = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "algorithm": algorithm,
            "hyperparameters": hyperparameters,
            "results": results,
            "data_info": data_info,
            "random_state": random_state,
            "training_time_seconds": training_time_seconds,
            "inference_time_ms": inference_time_ms,
        }
        self.metadata_log.append(record)

    def _key(self, meta: Dict[str, Any]) -> str:
        """Unique key for deduplication."""
        return (
            f"{meta['algorithm']}|"
            f"{json.dumps(meta['hyperparameters'], sort_keys=True)}|"
            f"{meta.get('random_state')}"
        )

    def save(self, path: Optional[str] = None) -> str:
        """Save metadata to JSON (merges with existing)."""
        if path is None:
            project_root = Path(__file__).parent.parent
            path = project_root / "model_metadata" / "experiments" / f"{self.experiment_name}.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing
        existing: List[Dict[str, Any]] = []
        if path.exists() and path.stat().st_size > 0:
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        
        # Merge
        merged = {self._key(m): m for m in existing}
        for m in self.metadata_log:
            merged[self._key(m)] = m
        
        path.write_text(json.dumps(list(merged.values()), indent=2))
        print(f"Metadata saved to {path}")
        
        return str(path)

    def best(self, metric: str = "test_accuracy") -> Optional[Dict[str, Any]]:
        """Get best experiment by metric."""
        if not self.metadata_log:
            return None
        return max(self.metadata_log, key=lambda r: r["results"].get(metric, float("-inf")))

    def summary(self) -> None:
        """Print summary of logged experiments."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {self.experiment_name} | Total runs: {len(self.metadata_log)}")
        print(f"{'='*70}")
        
        if not self.metadata_log:
            print("No experiments logged.")
            return
        
        print(f"{'Algorithm':<20} {'Accuracy':>10} {'F1':>10} {'Train(s)':>10} {'Infer(ms)':>10}")
        print("-" * 70)
        
        for r in self.metadata_log:
            acc = r["results"].get("test_accuracy", r["results"].get("accuracy"))
            f1 = r["results"].get("test_f1", r["results"].get("f1"))
            train_t = r.get("training_time_seconds")
            infer_t = r.get("inference_time_ms")
            
            print(
                f"{r['algorithm']:<20} "
                f"{acc:>10.4f}" if acc else f"{'N/A':>10}"
                f"{f1:>10.4f}" if f1 else f"{'N/A':>10}"
                f"{train_t:>10.2f}" if train_t else f"{'N/A':>10}"
                f"{infer_t:>10.4f}" if infer_t else f"{'N/A':>10}"
            )
        
        print(f"{'='*70}\n")

# Helper Func to measure time
import time
import numpy as np

def measure_inference_time(model, X_sample, n_runs: int = 100) -> float:
    """
    Measure average inference time per sample in milliseconds.
    
    Args:
        model: Trained model with .predict() method
        X_sample: Sample data (uses first 100 rows or all if smaller)
        n_runs: Number of runs to average
    
    Returns:
        Average inference time per sample in milliseconds
    """
    # Use subset for timing
    n_samples = min(100, X_sample.shape[0])
    X_subset = X_sample[:n_samples]
    
    # Warm-up run
    _ = model.predict(X_subset)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_subset)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_total = np.mean(times)
    avg_per_sample_ms = (avg_total / n_samples) * 1000
    
    return avg_per_sample_ms