from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from modules.track_metadata import ModelMetadata, measure_inference_time


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logger(name: str = "experiment_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a logger with console handler."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single SVM experiment."""
    name: str
    kernel: str
    param_space: Dict[str, tuple]
    n_trials: int = 30
    cv_folds: int = 5
    scoring: str = "accuracy"
    timeout: Optional[int] = None


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(kernel: str, params: dict, random_state: int = 42):
    """
    Create SVM model with given kernel and parameters.
    
    Args:
        kernel: 'linear', 'rbf', or 'poly'
        params: Model hyperparameters
        random_state: Random seed
    
    Returns:
        Configured SVM model (unfitted)
    """
    if kernel == "linear":
        return LinearSVC(
            **params,
            max_iter=1000000,
            tol=1e-4,        # Default tolerance
            dual="auto",
            random_state=random_state,
        )
    else:
        return SVC(
            kernel=kernel,
            **params,
            max_iter=-1,     # No limit for SVC (default)
            tol=1e-3,        # Slightly relaxed tolerancel
            random_state=random_state,
        )


# =============================================================================
# OPTUNA OPTIMIZATION
# =============================================================================

def create_optuna_objective(
    config: ExperimentConfig,
    X_train,
    y_train,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Args:
        config: Experiment configuration
        X_train, y_train: Training data
        random_state: Random seed
        n_jobs: Parallel jobs for CV
    
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        params = {}
        
        for param_name, param_config in config.param_space.items():
            suggest_type, *args = param_config
            
            if suggest_type == "float_log":
                params[param_name] = trial.suggest_float(param_name, args[0], args[1], log=True)
            elif suggest_type == "float":
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif suggest_type == "int":
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif suggest_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, args[0])
        
        model = create_model(config.kernel, params, random_state)
        
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=config.cv_folds,
            scoring=config.scoring,
            n_jobs=n_jobs,
        )
        
        return scores.mean()
    
    return objective


def run_optuna_optimization(
    config: ExperimentConfig,
    X_train,
    y_train,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[optuna.Study, float]:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        config: Experiment configuration
        X_train, y_train: Training data
        random_state: Random seed
        n_jobs: Parallel jobs
    
    Returns:
        (study, optimization_time_seconds)
    """
    logger.info(
        "Starting optimization: %s (%d trials, %d-fold CV)",
        config.name, config.n_trials, config.cv_folds
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction="maximize",
        study_name=config.name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    objective = create_optuna_objective(config, X_train, y_train, random_state, n_jobs)
    
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        show_progress_bar=True,
    )
    optimization_time = time.time() - start_time
    
    logger.info("Best params: %s", study.best_params)
    logger.info("Best CV score: %.4f", study.best_value)
    logger.info("Optimization time: %.1fs", optimization_time)
    
    return study, optimization_time


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_model(model, X_train, y_train) -> Tuple[Any, float]:
    """
    Train model and measure training time.
    
    Returns:
        (fitted_model, training_time_seconds)
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info("Training completed in %.2fs", training_time)
    
    return model, training_time


def evaluate_model(
    model,
    X_test,
    y_test,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with accuracy, f1, predictions, confusion_matrix
    """
    y_pred = model.predict(X_test)
    
    results = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    
    logger.info("Test Accuracy: %.4f", results["test_accuracy"])
    logger.info("Test F1: %.4f", results["test_f1"])
    
    if class_names:
        report = classification_report(y_test, y_pred, target_names=class_names)
        logger.debug("Classification Report:\n%s", report)
    
    return results


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42,
    n_jobs: int = -1,
    save_model: bool = True,
    models_dir: Path = None,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment: optimize, train, evaluate, save.
    
    Returns:
        Dictionary containing model, params, results, study
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", config.name)
    logger.info("Kernel: %s | Trials: %d | CV: %d-fold", config.kernel, config.n_trials, config.cv_folds)
    logger.info("=" * 60)
    
    # 1. Optimize hyperparameters
    study, optimization_time = run_optuna_optimization(
        config, X_train, y_train, random_state, n_jobs
    )
    
    # 2. Create best model
    best_model = create_model(config.kernel, study.best_params, random_state)
    
    # 3. Train
    best_model, training_time = train_model(best_model, X_train, y_train)
    
    # 4. Evaluate
    eval_results = evaluate_model(best_model, X_test, y_test, class_names)
    
    # 5. Measure inference time
    inference_time = measure_inference_time(best_model, X_test)
    logger.info("Inference time: %.4f ms/sample", inference_time)
    
    # 6. Save model
    if save_model and models_dir:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{config.name.lower()}_best.joblib"
        joblib.dump(best_model, model_path)
        logger.info("Model saved: %s", model_path)
    
    # Compile results
    experiment_result = {
        "model": best_model,
        "params": study.best_params,
        "study": study,
        "config": config,
        "results": {
            "test_accuracy": eval_results["test_accuracy"],
            "test_f1": eval_results["test_f1"],
            "best_cv_score": study.best_value,
            "optimization_time_seconds": optimization_time,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "n_trials": len(study.trials),
        },
        "y_pred": eval_results["y_pred"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "data_info": {
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "n_features": X_train.shape[1],
        },
    }
    
    return experiment_result


# =============================================================================
# RESULTS & COMPARISON
# =============================================================================

def results_to_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert experiment results to a pandas DataFrame.
    
    Args:
        results: Dictionary of {experiment_name: experiment_result}
    
    Returns:
        DataFrame with one row per experiment
    """
    rows = []
    
    for name, data in results.items():
        row = {
            "model": name,
            "kernel": data["config"].kernel,
            "test_accuracy": data["results"]["test_accuracy"],
            "test_f1": data["results"]["test_f1"],
            "cv_score": data["results"]["best_cv_score"],
            "training_time_s": data["results"]["training_time_seconds"],
            "inference_time_ms": data["results"]["inference_time_ms"],
            "optimization_time_s": data["results"]["optimization_time_seconds"],
            "n_trials": data["results"]["n_trials"],
            **data["params"],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("test_f1", ascending=False).reset_index(drop=True)
    
    return df


def print_comparison(results: Dict[str, Dict]) -> None:
    """Log formatted comparison of all experiments."""
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)
    
    header = (
        f"{'Model':<15} {'Kernel':<10} {'Accuracy':>10} {'F1':>10} "
        f"{'CV Score':>10} {'Train(s)':>10} {'Infer(ms)':>10}"
    )
    logger.info(header)
    logger.info("-" * 80)
    
    best_model = None
    best_f1 = 0
    
    for name, data in results.items():
        r = data["results"]
        kernel = data["config"].kernel
        
        row = (
            f"{name:<15} {kernel:<10} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} "
            f"{r['best_cv_score']:>10.4f} {r['training_time_seconds']:>10.2f} "
            f"{r['inference_time_ms']:>10.4f}"
        )
        logger.info(row)
        
        if r["test_f1"] > best_f1:
            best_f1 = r["test_f1"]
            best_model = name
    
    logger.info("-" * 80)
    logger.info("Best model: %s (F1=%.4f)", best_model, best_f1)
    logger.info("=" * 80)


def get_best_model(results: Dict[str, Dict], metric: str = "test_f1") -> Tuple[Any, str]:
    """
    Get the best model across all experiments.
    
    Returns:
        (model, model_name)
    """
    best_name = max(
        results.keys(),
        key=lambda k: results[k]["results"].get(metric, 0)
    )
    return results[best_name]["model"], best_name


# =============================================================================
# METADATA LOGGING
# =============================================================================

def log_results_to_metadata(
    results: Dict[str, Dict],
    experiment_name: str,
    random_state: int = 42,
) -> ModelMetadata:
    """
    Log all experiment results to ModelMetadata.
    
    Returns:
        ModelMetadata instance (already saved)
    """
    metadata = ModelMetadata(experiment_name)
    
    for name, data in results.items():
        metadata.log_experiment(
            algorithm=name,
            hyperparameters=data["params"],
            results={
                "test_accuracy": data["results"]["test_accuracy"],
                "test_f1": data["results"]["test_f1"],
                "best_cv_score": data["results"]["best_cv_score"],
            },
            data_info=data["data_info"],
            random_state=random_state,
            training_time_seconds=data["results"]["training_time_seconds"],
            inference_time_ms=data["results"]["inference_time_ms"],
        )
    
    metadata.save()
    logger.info("Metadata saved for experiment: %s", experiment_name)
    
    return metadata


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_experiment_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    experiments: List[ExperimentConfig],
    experiment_name: str = "svm_experiments",
    random_state: int = 42,
    n_jobs: int = -1,
    save_models: bool = True,
    class_names: List[str] = None,
) -> Dict[str, Dict]:
    """
    Run complete experiment pipeline.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        experiments: List of ExperimentConfig
        experiment_name: Name for metadata tracking
        random_state: Random seed
        n_jobs: Parallel jobs (-1 for all cores)
        save_models: Whether to save models to disk
        class_names: Names for classification report
    
    Returns:
        Dictionary of {experiment_name: experiment_result}
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "trained"
    
    logger.info("#" * 60)
    logger.info("EXPERIMENT PIPELINE: %s", experiment_name)
    logger.info("Total experiments: %d", len(experiments))
    logger.info("#" * 60)
    
    total_start = time.time()
    results = {}
    
    for config in experiments:
        result = run_single_experiment(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
            n_jobs=n_jobs,
            save_model=save_models,
            models_dir=models_dir,
            class_names=class_names,
        )
        results[config.name] = result
    
    total_time = time.time() - total_start
    
    log_results_to_metadata(results, experiment_name, random_state)
    print_comparison(results)
    
    logger.info("Total pipeline time: %.1f minutes", total_time / 60)
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_model(results: Dict[str, Dict], name: str):
    """Get a specific model by name."""
    if name not in results:
        raise KeyError(f"Model '{name}' not found. Available: {list(results.keys())}")
    return results[name]["model"]


def get_predictions(results: Dict[str, Dict], name: str) -> np.ndarray:
    """Get predictions for a specific model."""
    return results[name]["y_pred"]


def get_confusion_matrix(results: Dict[str, Dict], name: str) -> np.ndarray:
    """Get confusion matrix for a specific model."""
    return results[name]["confusion_matrix"]


def load_saved_model(model_name: str, models_dir: Path = None):
    """Load a saved model from disk."""
    if models_dir is None:
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models" / "trained"
    
    model_path = models_dir / f"{model_name.lower()}_best.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)