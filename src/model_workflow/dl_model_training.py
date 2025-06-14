# src/dl_model_training.py
import polars as pl # Still imported, but main usage is in data_preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras # For Keras/TF models logging
import mlflow.tensorflow # Also useful for TensorFlow specific logging/registration
from sklearn.model_selection import train_test_split # Not used here, done in data_preprocessing
from tensorflow.keras.models import Sequential, load_model
# from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification # Uncomment if you plan to reintroduce BERT/DistilBERT
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pickle, os, sys # Added sys for exit

# Library for model building
from tensorflow.keras.regularizers import l2
import keras
from keras.layers import SimpleRNN,LSTM,GRU, Embedding, Dense, SpatialDropout1D, Dropout, BatchNormalization, Bidirectional
# from attention import BahdanauAttention # Uncomment if you use custom Attention layer
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt # For KerasTuner
from tensorflow.keras.callbacks import EarlyStopping

# Library to overcome Warnings
import warnings
warnings.filterwarnings('ignore')

import logging

# SET UP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # log to console
        logging.FileHandler("training.log") # log to file
    ]
)
logger = logging.getLogger(__name__)

# Load preprocessed data
def load_data_for_training(data_dir, tokenizer_path):
    """
    Loads preprocessed numpy arrays (X_train, y_train, etc.) and tokenizer from specified paths.
    Args:
        data_dir (str): Directory containing X_*.npy, y_*.npy files.
        tokenizer_path (str): Path to tokenizer.pkl file.
    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test, tokenizer
    """
    logger.info(f"Loading data from directory: {data_dir}")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'), allow_pickle=True)

        y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
    except FileNotFoundError as e:
        logger.error(f"Error loading data files: {e}. Ensure data_preprocessing.py was run correctly and files are present.")
        sys.exit(1)

    logger.info(f"X train shape = {X_train.shape}")
    logger.info(f"y train shape = {y_train.shape}")
    logger.info(f"X test shape = {X_test.shape}")
    logger.info(f"y test shape = {y_test.shape}")
    logger.info(f"X val shape = {X_val.shape}")
    logger.info(f"y val shape = {y_val.shape}")

    # load the tokenizer
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info(f"Tokenizer loaded successfully from {tokenizer_path}!")
    except FileNotFoundError as e:
        logger.error(f"Error loading tokenizer: {e}. Ensure data_preprocessing.py was run correctly and tokenizer.pkl is present.")
        sys.exit(1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer

# --- Model Building and Training Functions ---

def build_and_train_bi_rnn_model(X_train, y_train, X_val, y_val, input_dim, max_len):
    """
    Builds and trains a Bidirectional SimpleRNN model.
    Logs model, parameters, and metrics to MLflow.
    """
    RNN_model = Sequential()
    RNN_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    RNN_model.add(SpatialDropout1D(0.2))
    RNN_model.add(Bidirectional(SimpleRNN(64, return_sequences=True)))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(BatchNormalization())
    RNN_model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(BatchNormalization())
    RNN_model.add(SimpleRNN(16, return_sequences=False)) # Last RNN layer returns sequences=False

    RNN_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    RNN_model.add(Dense(1, activation='sigmoid'))

    RNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    epochs = 15
    batch_size = 32
    logger.info(f"[INFO]: Starting training BiRNN model for {epochs} epochs with batch size {batch_size} ...")
    history = RNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
    logger.info(f"[INFO]: Finished training BiRNN model.")
    
    with mlflow.start_run(run_name="BiRNN_model_run", nested=True):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'embedding_output_dim': 100,
            'spatial_dropout': 0.2,
            'rnn_layer_1_units': 64,
            'rnn_layer_2_units': 32,
            'rnn_layer_3_units': 16, # Last SimpleRNN
            'dense_units': 64,
            'l2_reg': 0.01
        })
        # Log final validation accuracy and loss
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        
        mlflow.keras.log_model(RNN_model, 'BiRNN_model_artifact') # artifact_path
        logger.info("BiRNN model and parameters logged to MLflow.")

    return RNN_model, history

def build_and_train_lstm_model(X_train, y_train, X_val, y_val, input_dim, max_len):
    """
    Builds and trains an LSTM model.
    Logs model, parameters, and metrics to MLflow.
    """
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    lstm_model.add(SpatialDropout1D(0.5))
    lstm_model.add(LSTM(5, return_sequences=False)) # Return sequences=False for final LSTM
    lstm_model.add(Dropout(0.5))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 15 
    batch_size = 128
    logger.info(f"[INFO]: Starting training LSTM model for {epochs} epochs with batch size {batch_size} ...")
    history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
    logger.info(f"[INFO]: Finished training LSTM model.")
    
    with mlflow.start_run(run_name='LSTM_model_run', nested=True):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'embedding_output_dim': 100,
            'spatial_dropout': 0.5,
            'lstm_units': 5,
            'dense_units': 1
        })
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        
        mlflow.keras.log_model(lstm_model, 'LSTM_model_artifact')
        logger.info("LSTM model and parameters logged to MLflow.")
    
    return lstm_model, history

def build_and_train_gru_model(X_train, y_train, X_val, y_val, input_dim, max_len):
    """
    Builds and trains a GRU model.
    Logs model, parameters, and metrics to MLflow.
    """
    GRU_model = Sequential()
    GRU_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    GRU_model.add(SpatialDropout1D(0.5))
    GRU_model.add(GRU(5, return_sequences=False)) # Return sequences=False for final GRU
    GRU_model.add(Dropout(0.5))
    GRU_model.add(Dense(1, activation='sigmoid'))

    GRU_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    epochs = 20
    batch_size = 256
    logger.info(f"[INFO]: Starting training GRU model for {epochs} epochs with batch size {batch_size} ...")
    history = GRU_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
    logger.info(f"[INFO]: Finished training GRU model.")
    
    with mlflow.start_run(run_name='GRU_model_run', nested=True):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'embedding_output_dim': 100,
            'spatial_dropout': 0.5,
            'gru_units': 5,
            'dense_units': 1
        })
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        
        mlflow.keras.log_model(GRU_model, 'GRU_model_artifact')
        logger.info("GRU model and parameters logged to MLflow.")
    
    return GRU_model, history

def build_gru_model_fn(input_dim, input_length):
    """
    Returns a Keras Tuner model-building function for GRU, allowing hyperparameter optimization.
    """
    def model_builder(hp):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=input_length))

        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(rate=hp.Float('dropout_rate_1', 0.1, 0.5, step=0.1))) # Renamed for uniqueness

        # Allow for multiple GRU layers (up to 3 additional layers)
        for i in range(hp.Int('num_layers', 0, 2)): # 0 means 1 GRU layer (total 2), 2 means 3 extra layers (total 4 GRU layers)
            model.add(GRU(units, return_sequences=True)) # All intermediate layers return sequences
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i+2}', 0.1, 0.5, step=0.1))) # Unique name for each dropout

        model.add(GRU(units, return_sequences=False)) # Final GRU layer returns sequences=False
        model.add(Dropout(rate=hp.Float('final_dropout_rate', 0.1, 0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.Adam(
                        learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    return model_builder

def tune_and_log_gru(x_train, y_train, x_val, y_val, input_dim, input_length, project_name="gru_sentiment_tuning", model_registry_name="SentimentAnalysisModel"):
    """
    Performs hyperparameter tuning for GRU model using KerasTuner,
    logs results to MLflow, and registers the best model if criteria met.
    """
    num_max_trials = 10
    tuner = kt.RandomSearch(
        build_gru_model_fn(input_dim, input_length),
        objective="val_accuracy",
        max_trials=num_max_trials,
        executions_per_trial=1,
        directory="tuner_logs",
        project_name=project_name,
        overwrite=True # Set to False if you want to resume old searches
    )
    logger.info(f"[INFO]: Starting hyperparameter tuning for GRU model (max_trials={num_max_trials}) ...")
    
    # Define an early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    tuner.search(x_train, y_train,
                 validation_data=(x_val, y_val),
                 epochs=10, # Max epochs for tuner search
                 batch_size=32,
                 callbacks=[early_stopping],
                 verbose=2)
    logger.info(f"[INFO]: Finished hyperparameter tuning for GRU model.")

    best_models = tuner.get_best_models(num_models=1)
    if not best_models:
        logger.error("No best model found from KerasTuner search.")
        return None, None
    best_model = best_models[0]

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Ensure MLflow experiment is set for this tuning run
    mlflow.set_experiment("GRU Hypertuning")

    with mlflow.start_run(run_name='Hypertuned_GRU_Training_Run', nested=True) as run:
        logger.info(f"MLflow Run ID for Hypertuned GRU: {run.info.run_id}")
        
        # Log best hyperparameters
        for hp_name, hp_value in best_hps.values.items():
            mlflow.log_param(f"hp_{hp_name}", hp_value)

        # Retrain the best model with potentially more epochs
        logger.info("[INFO]: Retraining best GRU model with best hyperparameters...")
        retrain_epochs = 20 # Can be more epochs for final training
        history = best_model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=retrain_epochs,
            batch_size=32,
            callbacks=[early_stopping], # Apply early stopping again for final fit
            verbose=2
        )

        val_acc = history.history["val_accuracy"][-1]
        val_loss = history.history["val_loss"][-1]
        
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("final_val_loss", val_loss)
        mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("final_train_loss", history.history["loss"][-1])

        # Log and Register the best model to MLflow Model Registry
        ACCURACY_THRESHOLD = 0.75 # Example: only register if accuracy > 75%
        if val_acc >= ACCURACY_THRESHOLD:
            logger.info(f"Model accuracy ({val_acc:.4f}) meets threshold. Registering model '{model_registry_name}'...")
            # mlflow.tensorflow.log_model for Keras models
            mlflow.tensorflow.log_model(
                keras_model=best_model,
                artifact_path="hyper_gru_model", # This is where the model is saved within the MLflow artifact store
                registered_model_name=model_registry_name,
                # signature=infer_signature(...) # Optional: Define input/output signatures for better serving
                # input_example=X_train[0:1] # Optional: Example input for model serving
            )
            logger.info(f"Model '{model_registry_name}' registered successfully.")
        else:
            logger.info(f"Model accuracy ({val_acc:.4f}) below threshold {ACCURACY_THRESHOLD}. Not registering.")

        # Log best hyperparameters to a JSON file as an artifact
        best_hps_json_path = "best_hyperparams.json"
        with open(best_hps_json_path, "w") as f:
            import json
            json.dump(best_hps.values, f)
        mlflow.log_artifact(best_hps_json_path, artifact_path="hyperparameters")
        os.remove(best_hps_json_path) # Clean up local file

    return best_model, history # Return history from final fit for plotting

# --- Evaluation and Plotting Functions ---
def predict_and_log_metrics(model_name, model, X_test, y_test):
    """
    Evaluates a given model, logs metrics, classification report,
    and confusion matrix to MLflow.
    """
    logger.info(f"[INFO]: Evaluating {model_name} model on test set...")
    y_pred = model.predict(X_test, verbose=0) # Suppress verbose output for predict
    y_pred_binary = (y_pred > 0.5).astype('int')
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    logger.info(f"{model_name} Test Accuracy Score is: {accuracy*100:.2f}%")
    
    sentiment_labels = {0: 'negative', 1: 'positive'}
    
    # Use nested run for individual model evaluation metrics
    with mlflow.start_run(run_name=f'Metrics_of_{model_name}_run', nested=True) as run:
        # Log the primary evaluation metrics
        mlflow.log_metrics({
            'test_accuracy': accuracy,
            'test_precision': precision_score(y_test, y_pred_binary, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_pred_binary, average='weighted', zero_division=0),
            'test_f1_score': f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
        })

        # Classification Report
        report_dict = classification_report(y_test, y_pred_binary, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = f'classification_report_for_{model_name}.csv'
        report_df.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path, artifact_path='evaluation_reports')
        os.remove(report_path) # Clean up local file

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        # Using pandas for confusion matrix for broader compatibility in MLflow logging
        cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        
        cm_path = f'confusion_matrix_of_{model_name}.csv'
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path, artifact_path='evaluation_reports')
        os.remove(cm_path) # Clean up local file

        logger.info(f"Metrics and reports for {model_name} logged to MLflow (Run ID: {run.info.run_id}).")

    # Return sentiments for potential downstream use (e.g., sample prediction logging)
    model_sentiments = [[sentiment_labels[val[0]]] for val in y_pred_binary]
    return model_sentiments

def plot_model_history(model_name, history, plot_output_dir):
    """
    Plots training and validation accuracy and loss for a model and logs it as an MLflow artifact.
    """
    os.makedirs(plot_output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='upper left')

    # Plot loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(plot_output_dir, f'{model_name}_accuracy_loss.png')
    plt.savefig(plot_path)
    logger.info(f"[INFO]: Plot saved to {plot_path}")

    # Log plot as MLflow artifact within a nested run
    with mlflow.start_run(run_name=f'Plot_of_{model_name}', nested=True):
        mlflow.log_artifact(plot_path, artifact_path="model_performance_plots")
        logger.info(f"Plot for {model_name} logged as MLflow artifact.")
    
    plt.close(fig) # Close the plot to free memory

# --- Main Orchestrating Function for Training and Evaluation ---
def train_and_evaluate_all_models(data_dir, tokenizer_path, plot_output_dir, model_registry_name="SentimentAnalysisModel"):
    """
    Orchestrates the training and evaluation of all defined models.
    Args:
        data_dir (str): Directory containing X_*.npy, y_*.npy files.
        tokenizer_path (str): Path to tokenizer.pkl file.
        plot_output_dir (str): Directory to save plots.
        model_registry_name (str): Name to use for MLflow Model Registry.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = load_data_for_training(data_dir, tokenizer_path)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_mlflow_uri = f"file://{os.path.join(project_root, 'mlruns')}"
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_mlflow_uri)
    # Get vocabulary size and max_len for embedding layer
    input_dim = len(tokenizer.word_index) + 1 # Vocabulary size + 1 for padding
    max_len = X_train.shape[1] # Length of padded sequences
    logger.info(f"Embedding layer input_dim (vocab_size): {input_dim}")
    logger.info(f"Embedding layer input_length (max_len): {max_len}")

    # Set MLflow Tracking URI from environment variable (crucial for CI/CD/Kaggle)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment("Sentiment Analysis Models") # Top-level experiment

    # List of models to train and evaluate
    models_to_train = {
        "BiRNN": build_and_train_bi_rnn_model,
        "LSTM": build_and_train_lstm_model,
        # "GRU": build_and_train_gru_model
    }

    trained_models = {}
    model_histories = {}

    # Train and evaluate standard models
    for model_name, build_fn in models_to_train.items():
        try:
            model, history = build_fn(X_train, y_train, X_val, y_val, input_dim, max_len)
            trained_models[model_name] = model
            model_histories[model_name] = history # Store history for plotting
            
            predict_and_log_metrics(model_name, model, X_test, y_test)
            plot_model_history(model_name, history, plot_output_dir)
            
            # Log the best metric of this model type to the main experiment for comparison
            # Note: This is an example, typically you'd log key metrics to the overall experiment run
            # or use MLflow's compare runs feature.
            # with mlflow.start_run(run_name=f"Summary_{model_name}", nested=True):
            #     mlflow.log_metric(f"{model_name}_val_accuracy", history.history['val_accuracy'][-1])
            #     mlflow.log_metric(f"{model_name}_test_accuracy", accuracy_score((model.predict(X_test) > 0.5).astype('int'), y_test))

        except Exception as e:
            logger.error(f"Error training/evaluating {model_name} model: {e}", exc_info=True)
            # Decide if pipeline should exit or continue with other models.
            # For robustness, we'll continue but log the error.

    # Hyperparameter tuning and logging for GRU
    logger.info("Starting Hyperparameter Tuning for GRU model...")
    best_gru_model, best_gru_history = tune_and_log_gru( # Now returns history for plotting
        X_train, y_train, X_val, y_val, 
        input_dim=input_dim, 
        input_length=max_len,
        model_registry_name=model_registry_name # Pass the registry name
    )

    if best_gru_model:
        trained_models['Hypertuned_GRU'] = best_gru_model
        model_histories['Hypertuned_GRU'] = best_gru_history # Store history
        predict_and_log_metrics('Hypertuned_GRU', best_gru_model, X_test, y_test)
        plot_model_history('Hypertuned_GRU', best_gru_history, plot_output_dir) # Plot history of best model
    
    logger.info("All model training and evaluation cycles completed.")

if __name__ == '__main__':
    # For local testing, ensure dummy data and tokenizer are present
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir_local = os.path.join(project_root, 'data')
    tokenizer_path_local = os.path.join(project_root, 'data', 'tokenizer.pkl')
    plot_output_dir_local = os.path.join(project_root, 'plots')
    
    # Ensure a dummy tokenizer.pkl and dummy .npy files exist for standalone testing
    if not os.path.exists(tokenizer_path_local) or \
       not os.path.exists(os.path.join(data_dir_local, 'X_train.npy')):
        logger.warning("No preprocessed data found. Attempting to run data_preprocessing.py to generate it...")
        # Import and run the preprocessing script's main function
        from src.data_workflow.data_preprocessing import validate_and_preprocess_data_ge
        raw_data_path_local_dp = os.path.join(project_root, 'data', 'IMDB-Dataset.csv')
        
        # Create a dummy IMDB-Dataset.csv if it doesn't exist for local testing
        if not os.path.exists(raw_data_path_local_dp):
            print(f"Creating dummy raw data at {raw_data_path_local_dp}")
            dummy_data = {
                'review': [
                    "This movie was absolutely fantastic! I loved every moment of it. Highly recommend.",
                    "Terrible experience, utterly disappointed. Waste of time and money. :(",
                    "It's an okay film, nothing special. Could be better, could be worse. meh",
                    "A truly great and inspiring movie. Loved the acting and the plot. :)",
                    "Worst movie ever. I'm so angry I watched it. #fail"
                ],
                'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative']
            }
            pd.DataFrame(dummy_data).to_csv(raw_data_path_local_dp, index=False)
            print("Dummy raw data created for data_preprocessing.py.")
        
        validate_and_preprocess_data_ge(raw_data_path_local_dp, data_dir_local, tokenizer_path_local)

    # Set a dummy MLflow Tracking URI for local testing if not already set globally
    # This will be overridden by environment variables in CI/CD/Kaggle
    os.environ['MLFLOW_TRACKING_URI'] = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:8080')
    
    train_and_evaluate_all_models(data_dir_local, tokenizer_path_local, plot_output_dir_local)