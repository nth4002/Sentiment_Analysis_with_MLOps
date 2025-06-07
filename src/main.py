# src/main.py
import os
import sys
import subprocess
import shutil

# Import the main functions from your modified scripts
from data_workflow.data_preprocessing import validate_and_preprocess_data_ge
from model_workflow.dl_model_training import train_and_evaluate_all_models

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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(project_root)
    raw_data_path = os.path.join(project_root, 'data2', 'dummy_data.csv') # Your raw data file name
    print(raw_data_path)
    processed_data_output_dir = os.path.join(project_root, 'data2') # Where to save .npy files
    tokenizer_path = os.path.join(project_root, 'data2', 'tokenizer.pkl')
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
    # logger.info("\n--- DVC Adding/Pushing Processed Data and Tokenizer ---")
    # Add .dvc files for the generated processed data and tokenizer
    run_dvc_command(["add", os.path.join('data2', 'X_train.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'X_test.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'X_val.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'y_train.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'y_test.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'y_val.npy')], cwd=project_root)
    run_dvc_command(["add", os.path.join('data2', 'tokenizer.pkl')], cwd=project_root)

    # Commit changes to Git for the .dvc files
     # Commit changes to Git for the .dvc files
    try:
        # Run git config commands,python's subprocess will capture stdout/stderr 
        subprocess.run(
            ["git", "config", "user.name", "github-actions[bot]"], 
            check=True, cwd=project_root
        )
        subprocess.run(
            ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
            check=True, cwd=project_root
        )

        # Correctly add all .dvc files in the data directory.
        # We can pass the directory itself, or use a shell to expand the wildcard.
        # Passing the directory is safer and cleaner.
        dvc_files_dir = os.path.join('data2', '') # Path to the directory containing .dvc files
        subprocess.run(
            ["git", "add", dvc_files_dir], # 'git add data/' will stage all new/modified files in 'data'
            cwd=project_root, check=True
        )

        # Clean up the git commit command.
        # The logic for --no-verify is fine.
        commit_message = "feat(data): Update processed data and tokenizer via automated pipeline"
        commit_command = ["git", "commit", "-m", commit_message]
        if os.getenv('CI'):
            commit_command.append('--no-verify')
            
        # check=False is good here, as it prevents failure if there's nothing to commit.
        subprocess.run(commit_command, cwd=project_root, check=False)
        logger.info("Git commit for processed data DVC files complete (if changes were present).")

    except subprocess.CalledProcessError as e:
        # Capture stderr for better error logging if available
        stderr_output = e.stderr.decode() if e.stderr else "No stderr output."
        logger.warning(f"A git command failed with exit code {e.returncode}: {stderr_output}")
    
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

    