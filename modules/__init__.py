"""
Experiment tracking and pipeline modules for ML experiments.
"""

from .track_metadata import (
    ModelMetadata,
    measure_inference_time,
)

from .experiment_pipeline import (
    # Configuration
    ExperimentConfig,
    
    # Model creation
    create_model,
    
    # Optuna optimization
    create_optuna_objective,
    run_optuna_optimization,
    
    # Training & evaluation
    train_model,
    evaluate_model,
    
    # Single experiment
    run_single_experiment,
    
    # Results & comparison
    results_to_dataframe,
    print_comparison,
    get_best_model,
    
    # Metadata logging
    log_results_to_metadata,
    
    # Main pipeline
    run_experiment_pipeline,
    
    # Convenience functions
    get_model,
    get_predictions,
    get_confusion_matrix,
    load_saved_model,
    
    # Logger
    setup_logger,
)

__all__ = [
    # track_metadata
    "ModelMetadata",
    "measure_inference_time",
    
    # experiment_pipeline
    "ExperimentConfig",
    "create_model",
    "create_optuna_objective",
    "run_optuna_optimization",
    "train_model",
    "evaluate_model",
    "run_single_experiment",
    "results_to_dataframe",
    "print_comparison",
    "get_best_model",
    "log_results_to_metadata",
    "run_experiment_pipeline",
    "get_model",
    "get_predictions",
    "get_confusion_matrix",
    "load_saved_model",
    "setup_logger",
]