# src/main.py
import os
import sys
import subprocess
import shutil

# Import the main functions from your modified scripts
from src.data_preprocessing import validate_and_preprocess_data_ge
from src.dl_model_training import train_and_evaluate_all_models

# Configure logging for the main orchestrator
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline_orchestrator.log")]
)
logger = logging.getLogger(__name__)

def run_dvc_command(command_args, cwd=None):
    """Helper function to run DVC commands."""
    cmd = ["dvc"] + command_args
    logger.info(f"Running DVC command: {' '.join(cmd)}")
    try:
        # Use check=True to raise CalledProcessError if the command fails
        result = subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr: # DVC often logs non-critical messages to stderr
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {' '.join(cmd)}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        sys.exit(1) # Stop the pipeline on DVC failure
    except FileNotFoundError:
        logger.error("DVC command not found. Is DVC installed and in your PATH?")
        sys.exit(1)

def main():
    # Define paths relative to the project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    raw_data_path = os.path.join(project_root, 'data', 'IMDB-Dataset.csv') # Your raw data file name
    processed_data_output_dir = os.path.join(project_root, 'data') # Where to save .npy files
    tokenizer_path = os.path.join(project_root, 'data', 'tokenizer.pkl')
    plot_output_dir = os.path.join(project_root, 'plots') # Directory for saving plots locally
    model_registry_name = "SentimentAnalysisModel" # Name for MLflow Model Registry

    # Ensure necessary directories exist
    os.makedirs(processed_data_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    logger.info("\n--- Starting MLOps Training Pipeline ---")

    # STEP 1: Data Pull (DVC)
    logger.info("\n--- DVC Pulling Data ---")
    # This pulls raw_data.csv and any previously DVC-tracked processed data/tokenizer.
    # This ensures the pipeline always starts with the latest available data artifacts.
    run_dvc_command(["pull"], cwd=project_root)
    logger.info("DVC pull successful.")

    # STEP 2: Data Validation & Preprocessing
    logger.info("\n--- Running Data Validation and Preprocessing ---")
    try:
        validate_and_preprocess_data_ge(raw_data_path, processed_data_output_dir, tokenizer_path)
        logger.info("Data Validation and Preprocessing successful.")
    except SystemExit: # Catch the sys.exit(1) from GE if validation fails
        logger.error("Pipeline stopped due to data validation failure in preprocessing.")
        sys.exit(1) # Propagate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred during data preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # STEP 2.5: DVC Add/Push Processed Data and Tokenizer
    # This will version and push the *outputs* of the preprocessing step.
    logger.info("\n--- DVC Adding/Pushing Processed Data and Tokenizer ---")
    # Add .dvc files for the generated processed data and tokenizer
    run_dvc_command(["add", os.path.join('data', 'X_train.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'X_test.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'X_val.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'y_train.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'y_test.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'y_val.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data', 'tokenizer.pkl')], cwd=project_root)

    # Commit changes to Git for the .dvc files
    try:
        subprocess.run(["git", "config", "2>&1", "user.name", "'github-actions[bot]'"], check=True, cwd=project_root)
        subprocess.run(["git", "config", "2>&1", "user.email", "'github-actions[bot]@users.noreply.github.com'"], check=True, cwd=project_root)
        subprocess.run(["git", "add", os.path.join('data', '*.dvc')], cwd=project_root, check=True)
        # Use || true to make commit non-failing if there's nothing to commit
        subprocess.run(["git", "commit", "-m", "feat(data): Update processed data and tokenizer via automated pipeline"] + (['--no-verify'] if os.getenv('CI') else []), cwd=project_root, check=False)
        logger.info("Git commit for processed data DVC files complete (if changes were present).")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git commit for DVC files failed: {e.stderr}. This might be normal if no changes.")
    
    # Push the actual data content to remote DVC storage
    run_dvc_command(["push"], cwd=project_root)
    logger.info("DVC push of processed data and tokenizer successful.")

    # STEP 3: Model Training, Evaluation, and Registration (MLflow)
    logger.info("\n--- Running Model Training and Evaluation ---")
    try:
        train_and_evaluate_all_models(processed_data_output_dir, tokenizer_path, plot_output_dir, model_registry_name)
        logger.info("Model Training, Evaluation, and Registration successful.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n--- MLOps Training Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()