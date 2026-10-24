import gc
import tensorflow as tf
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
import joblib
import logging
import psutil
import os
import json
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Disable oneDNN optimizations to suppress related warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging to include a file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_logs.log"),  # Log to a file
        logging.StreamHandler()                      # Also log to console
    ]
)

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

@contextmanager
def manage_resources():
    """Enhanced context manager for both memory and GPU resources"""
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield
            # Log FitFailedWarnings
            for warning in w:
                if issubclass(warning.category, FitFailedWarning):
                    logger.warning(f"Fit failed warning: {warning.message}")
    finally:
        # Clear memory
        gc.collect()
        
        # Clear GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass

def check_resources():
    """Check current resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    gpu_memory = []
    
    if tf.config.list_physical_devices('GPU'):
        try:
            gpu_memory = [tf.config.experimental.get_memory_info(device)['current'] 
                         for device in tf.config.list_physical_devices('GPU')]
        except:
            gpu_memory = ['Unknown']
            
    return {
        'cpu_percent': cpu_percent,
        'memory_mb': memory,
        'gpu_memory': gpu_memory
    }

def load_data():
    """Load training data"""
    train = pd.read_csv('output/cleaned_train.csv')
    X = train.drop(columns=['PremiumAmount'])
    y = train['PremiumAmount']
    return X, y

def safe_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error safely"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def save_model(model_info: dict, name: str, output_dir: Path):
    """Save model info to disk"""
    output_dir = Path(output_dir)
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the entire model_info dictionary
    model_path = model_dir / f'best_{name}_model.pkl'  # Ensure consistent naming
    try:
        joblib.dump(model_info, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model {name}: {str(e)}")
        raise

def load_model(name: str, output_dir: Path):
    """Load model from versioned directory.

    Supports both scikit-learn and Keras models.
    """
    model_filename = f'best_{name}_model.pkl'
    model_path = output_dir / name / model_filename  # Adjusted path to include model directory
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if name in ['NeuralNet']:
        # Load Keras model
        model = tf.keras.models.load_model(str(model_path), custom_objects={'safe_rmsle': safe_rmsle})
    else:
        # Load scikit-learn model
        model = joblib.load(str(model_path))
    
    logger.info(f"Loaded model '{name}' from {model_path}")
    return model

def load_experiment_state(state_path: Path) -> dict:
    """Load the experiment state from JSON."""
    with open(state_path, 'r') as f:
        state = json.load(f)
    return state

def save_state(output_dir, completed_experiments, failed_experiments, crashed_experiments):
    """Save experiment state.
    
    Args:
        completed_experiments: list of dicts with 'name' and 'score'
        failed_experiments: list of tuples (name, error)
        crashed_experiments: list of experiment names (strings)
    """
    # Ensure all items in completed_experiments are dictionaries
    cleaned_completed = []
    for exp in completed_experiments:
        if isinstance(exp, dict) and 'name' in exp and 'score' in exp:
            cleaned_completed.append({
                "name": exp['name'],
                "score": exp.get('score', None)
            })
        else:
            logger.warning(f"Skipping invalid completed experiment entry: {exp}")
    
    state_data = {
        "completed": cleaned_completed,
        "failed": failed_experiments,
        "crashed": list(crashed_experiments),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_dir / "experiment_state.json", "w") as f:
        json.dump(state_data, f, indent=2)

def load_state(output_dir: Path) -> dict:
    """Load experiment state if exists"""
    state_file = output_dir / 'experiment_state.json'
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'crashed': [], 'timestamp': None}

def generate_performance_report(results: list, best_models: dict) -> dict:
    """Generate detailed performance report for all models"""
    report = {
        'summary': [],
        'detailed': {}
    }
    
    # Collect all valid results
    valid_results = [r for r in results if r['status'] == 'success']
    
    if not valid_results:
        return {
            'error': 'No successful experiments to report',
            'summary': [],
            'detailed': {}
        }
    
    # Calculate best score for normalization
    best_score = min(r['score'] for r in valid_results)
    
    # Generate detailed stats for each model
    for result in valid_results:
        model_name = result.get('model').__class__.__name__
        score = result['score']
        
        # Calculate relative performance
        relative_performance = best_score / score if score > 0 else 0
        
        model_stats = {
            'score': score,
            'relative_performance': relative_performance,
            'status': result['status'],
            'features_used': len(result['features']),
            'error': result.get('error', None)
        }
        
        report['detailed'][model_name] = model_stats
        report['summary'].append({
            'model': model_name,
            'score': score,
            'relative_performance': relative_performance
        })
    
    # Handle non-finite scores
    non_finite = [r for r in results if not np.isfinite(r['score']) and r['status'] == 'success']
    if non_finite:
        report['warnings'] = [
            f"Model {r.get('model').__class__.__name__} has non-finite score: {r['score']}"
            for r in non_finite
        ]
        logger.warning("Some models have non-finite scores.")

    # Sort summary by score
    report['summary'].sort(key=lambda x: x['score'])
    
    # Add metadata
    report['metadata'] = {
        'total_experiments': len(results),
        'successful_experiments': len(valid_results),
        'best_score': best_score,
        'timestamp': datetime.now().isoformat()
    }
    
    return report

def log_memory_usage(stage: str):
    """Log current memory usage in GB."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)
    logger.info(f"{stage} - Current memory usage: {mem_usage:.2f} GB")

class FeatureNamePreserver(BaseEstimator, TransformerMixin):
    """Preserves feature names through sklearn pipeline"""
    def fit(self, X, y=None):
        self.feature_names_ = X.columns.tolist() if hasattr(X, 'columns') else None
        return self
        
    def transform(self, X):
        return X
        
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

def find_most_similar(test_record, train_data, n_neighbors=50):
    """Find the most similar training records to the test record"""
    scaler = StandardScaler()
    combined = pd.concat([train_data, test_record.to_frame().T], ignore_index=True)
    scaler.fit(combined)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_record.to_frame().T)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(train_scaled)
    distances, indices = nbrs.kneighbors(test_scaled)
    return train_data.iloc[indices[0]]
