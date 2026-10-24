# filepath: /home/USER/Kaggle/InsuranceNext/experiments/base_experiment.py
from abc import ABC, abstractmethod
from pathlib import Path
from utils import safe_rmsle, save_model, load_data as utils_load_data, log_memory_usage  # Ensure log_memory_usage is imported
from features import feature_importances  # Import feature_importances
import logging
import traceback
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """Load data with logging"""
        self.logger.info(f"Loading data for {self.name}")
        return utils_load_data()

    def log_memory_usage(self, stage: str):
        """Wrapper for the log_memory_usage function."""
        log_memory_usage(stage)

    def run(self) -> dict:
        """Template method pattern with enhanced error handling and memory logging"""
        try:
            self.log_memory_usage(f"{self.name} - Before loading data")
            X, y = self.load_data()
            self.log_memory_usage(f"{self.name} - After loading data")

            self.logger.info(f"Creating model for {self.name}")
            self.log_memory_usage(f"{self.name} - Before creating model")
            model = self.create_model()
            self.log_memory_usage(f"{self.name} - After creating model")

            self.logger.info(f"Training {self.name}")
            self.log_memory_usage(f"{self.name} - Before training model")
            model.fit(X, y)
            self.log_memory_usage(f"{self.name} - After training model")

            self.logger.info(f"Evaluating {self.name}")
            predictions = model.predict(X)
            score = safe_rmsle(y, predictions)
            self.log_memory_usage(f"{self.name} - After evaluation")
            self.logger.info(f"{self.name} score: {score:.4f}")

            model_info = self.create_model_info(model, X, y, score)

            try:
                save_model(model_info, self.name, self.output_dir)
                self.log_memory_usage(f"{self.name} - After saving model")
            except Exception as e:
                self.logger.error(f"Failed to save {self.name} model: {str(e)}")
                model_info['status'] = 'success_but_not_saved'
                model_info['error'] = str(e)
                self.log_memory_usage(f"{self.name} - After failed model save")

            return model_info

        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self.log_memory_usage(f"{self.name} - On failure")
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the experiment"""
        pass

    @abstractmethod
    def create_model(self):
        """Create and configure the model"""
        pass

    def create_model_info(self, model, X, y, score, additional_info=None):
        """Create standardized model info dictionary"""
        info = {
            'name': self.name,
            'model': model,
            'features': X.columns.tolist(),
            'score': score,
            'status': 'success',
            'error': None,
            'version': str(self.output_dir),
            'sample_size': len(X),
            'feature_count': len(X.columns),
            'timestamp': datetime.now().isoformat(),
            'parameters': getattr(model, 'get_params', lambda: {})()
        }

        if additional_info:
            info.update(additional_info)

        return info