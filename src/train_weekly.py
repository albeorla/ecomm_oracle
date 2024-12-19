#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger

from models.random_forest_model import RandomForestModel
from models.pipeline import MLPipeline

def run_command(cmd: list, error_msg: str) -> None:
    """Run a shell command and handle errors."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_msg}: {e.stderr}")
        raise e

def is_git_tracked(file_path: Path) -> bool:
    """Check if a file is tracked by Git."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '--error-unmatch', str(file_path)],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def setup_dvc():
    """Setup DVC if not already configured."""
    if not Path('.dvc').exists():
        logger.info("Initializing DVC")
        run_command(['dvc', 'init'], "Failed to initialize DVC")
        
        # Setup default remote storage
        dvc_storage = Path.home() / 'dvc-storage'
        dvc_storage.mkdir(exist_ok=True)
        run_command(
            ['dvc', 'remote', 'add', '-d', 'mystorage', str(dvc_storage)],
            "Failed to add DVC remote"
        )
        
        # Add DVC files to Git
        run_command(['git', 'add', '.dvc', '.dvcignore'], "Failed to add DVC files to Git")
        run_command(
            ['git', 'commit', '-m', "Initialize DVC"],
            "Failed to commit DVC initialization"
        )

def track_data_file(file_path: Path) -> None:
    """Track a data file with DVC, removing from Git if necessary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Check if file is tracked by Git
    if is_git_tracked(file_path):
        logger.info(f"Removing {file_path} from Git tracking")
        try:
            # Remove from Git tracking but keep the file
            run_command(
                ['git', 'rm', '--cached', str(file_path)],
                f"Failed to remove {file_path} from Git"
            )
            # Commit the change
            run_command(
                ['git', 'commit', '-m', f"stop tracking {file_path} in Git, moving to DVC"],
                f"Failed to commit removal of {file_path} from Git"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove file from Git: {e}")
            raise
    
    # Add file to DVC
    logger.info(f"Adding {file_path} to DVC tracking")
    run_command(['dvc', 'add', str(file_path)], f"Failed to add {file_path} to DVC")
    
    # Add the .dvc file and .gitignore to Git
    dvc_file = Path(f"{file_path}.dvc")
    if dvc_file.exists():
        run_command(
            ['git', 'add', str(dvc_file), str(file_path.parent / '.gitignore')],
            f"Failed to add {dvc_file} to Git"
        )

def train_weekly():
    """Run weekly model training pipeline with DVC data versioning."""
    # Ensure DVC is setup
    setup_dvc()
    
    # Setup paths
    data_dir = Path("data")
    history_dir = data_dir / "history"
    current_data = data_dir / "envelopes_products.csv"
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not current_data.exists():
        logger.error(f"Training data not found at {current_data}")
        raise FileNotFoundError(f"Training data not found at {current_data}")
    
    # Track current data with DVC
    track_data_file(current_data)
    
    # Archive current data
    logger.info(f"Archiving current data to history")
    history_dir.mkdir(exist_ok=True)
    archived_path = history_dir / f"envelopes_products_{timestamp}.csv"
    
    # Copy instead of move to keep DVC tracking
    import shutil
    shutil.copy2(current_data, archived_path)
    
    # Track archived data
    track_data_file(archived_path)
    
    # Use archived data as validation set
    val_data_path = str(archived_path)
    
    # Initialize model and pipeline
    model = RandomForestModel()
    pipeline = MLPipeline(
        model=model,
        experiment_name="product_profitability",
        run_name=f"weekly_training_{timestamp}"
    )
    
    try:
        # Run pipeline
        results = pipeline.run_pipeline(
            train_data_path=str(current_data),
            val_data_path=val_data_path,
            hyperparameter_tune=True,
            timestamp=timestamp
        )
        
        # Log results
        logger.info(f"Training metrics: {results['train_metrics']}")
        if results['val_metrics']:
            logger.info(f"Validation metrics: {results['val_metrics']}")
        logger.info(f"Model saved to: {results['model_path']}")
        
        # Commit DVC changes
        run_command(['dvc', 'commit'], "Failed to commit DVC changes")
        
        # Create a meaningful commit message
        metrics_summary = f"Train MAE: {results['train_metrics'].get('mae', 'N/A')}"
        if results['val_metrics']:
            metrics_summary += f", Val MAE: {results['val_metrics'].get('mae', 'N/A')}"
        
        commit_message = f"""Update data and model for training run {timestamp}

Metrics:
{metrics_summary}

Model path: {results['model_path']}
MLflow run ID: {results['run_id']}"""
        
        # Stage and commit DVC files
        try:
            run_command(
                ['git', 'add', '*.dvc', 'data/.gitignore'],
                "Failed to stage DVC files"
            )
            
            # Create a temporary file for the commit message
            msg_file = Path('/tmp/commit_msg.txt')
            msg_file.write_text(commit_message)
            
            run_command(
                ['git', 'commit', '-F', str(msg_file)],
                "Failed to commit DVC files"
            )
            msg_file.unlink()  # Clean up
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit failed: {e}. You may need to commit the changes manually.")
        
    except Exception as e:
        logger.error(f"Pipeline run failed: {e}")
        raise e

if __name__ == "__main__":
    train_weekly() 