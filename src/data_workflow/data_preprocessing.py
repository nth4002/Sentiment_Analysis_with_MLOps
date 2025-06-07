# src/data_preprocessing.py
import polars as pl
import numpy as np
import pandas as pd # pandas is needed for Great Expectations DataFrame-based validation and Keras tokenization
import re, sys
import os # Added for path manipulation

# preprocessing libraries
import contractions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import spacy
import pickle
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

# Great Expectations libraries
import great_expectations as gx # Use the gx alias

import logging
from logging import FileHandler # Explicitly import FileHandler

# SET UP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # log to console
        FileHandler("preprocessing.log") # log to file
    ]
)
logger = logging.getLogger(__name__)

# --- NLTK and SpaCy Resource Downloads ---
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resources = {
        "stopwords": 'corpora/stopwords',
        'punkt': 'tokenizers/punkt'
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK '{resource_name}' already downloaded!")
        except LookupError:
            logger.info(f"NLTK '{resource_name}' not found. Downloading...")
            nltk.download(resource_name, quiet=True)
            logger.info(f"NLTK resource '{resource_name}' downloaded successfully!")

download_nltk_resources() # Call once at the start

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info(f"SpaCy model loaded successfully!")
except OSError:
    logger.error(f"SpaCy model 'en_core_web_sm' not found. Please download it by running: python -m spacy download en_core_web_sm")
    sys.exit(1) # Exit if essential model is missing

# --- Chat Words Dictionary (unchanged) ---
chat_words = {
    "AFAIK": "As Far As I Know", "AFK": "Away From Keyboard", "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard", "ATM": "At The Moment", "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard", "BBL": "Be Back Later", "BBS": "Be Back Soon",
    "BFN": "Bye For Now", "B4N": "Bye For Now", "BRB": "Be Right Back",
    "BRT": "Be Right There", "BTW": "By The Way", "B4": "Before",
    "CU": "See You", "CUL8R": "See You Later", "CYA": "See You",
    "FAQ": "Frequently Asked Questions", "FC": "Fingers Crossed", "FWIW": "For What It's Worth",
    "FYI": "For Your Information", "GAL": "Get A Life", "GG": "Good Game",
    "GN": "Good Night", "GMTA": "Great Minds Think Alike", "GR8": "Great!",
    "G9": "Genius", "IC": "I See", "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You", "IMHO": "In My Honest/Humble Opinion", "IMO": "In My Opinion",
    "IOW": "In Other Words", "IRL": "In Real Life", "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship", "LMAO": "Laugh My A.. Off", "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See", "L8R": "Later", "MTE": "My Thoughts Exactly",
    "M8": "Mate", "NRN": "No Reply Necessary", "OIC": "Oh I See",
    "PITA": "Pain In The A..", "PRT": "Party", "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?", "ROFL": "Rolling On The Floor Laughing", "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off", "SK8": "Skate",
    "STATS": "Your sex and age", "ASL": "Age, Sex, Location", "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!", "TTYL": "Talk To You Later", "U": "You",
    "U2": "You Too", "U4E": "Yours For Ever", "WB": "Welcome Back",
    "WTF": "What The F...", "WTG": "Way To Go!", "WUF": "Where Are You From?",
    "W8": "Wait...", "7K": "Sick:-D Laughter", "TFW": "That feeling when",
    "MFW": "My face when", "MRW": "My reaction when", "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh", "JK": "Just kidding", "ILY": "I love you",
    "IMU": "I miss you", "ADIH": "Another day in hell", "IDC": "I don’t care",
    "ZZZ": "Sleeping, bored, tired", "WYWH": "Wish you were here", "TIME": "Tears in my eyes",
    "BAE": "Before anyone else", "FIMH": "Forever in my heart", "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter", "BFF": "Best friends forever", "CSL": "Can’t stop laughing",
}

# Added error handling for stopwords download just in case the first try fails
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    logger.warning("NLTK stopwords not found. Attempting download...")
    nltk.download("stopwords", quiet=True)
    try:
        STOP_WORDS = set(stopwords.words('english'))
        logger.info("NLTK stopwords downloaded and loaded.")
    except LookupError:
        logger.error("Failed to download NLTK stopwords. Exiting.")
        sys.exit(1)


# --- Data Loading and Initial Info/Deduplication ---
def dataset_info(df_polars):
    """
    Provides an overview of the dataset and handles duplicates.
    Accepts a Polars DataFrame and returns a Polars DataFrame.
    """
    logger.info(f"Dataset shape = {df_polars.shape}")
    logger.info(f"Overview of dataset: \nSchema: {df_polars.schema}\nNull Count: \n{df_polars.null_count()}\n")

    num_duplicates = df_polars.is_duplicated().sum()
    logger.info(f"Checking duplicated records: {num_duplicates}")

    if num_duplicates > 0:
        logger.info(f"Perform Drop Duplicates: \n")
        df_polars = df_polars.unique()
        logger.info(f"After dropping duplicates, data shape: {df_polars.shape}")
    else:
        logger.info(f"No duplicate records found.")

    logger.info(f"Finished dataset info checked!")
    return df_polars

# --- Text Preprocessing Functions (mostly unchanged, slight improvements) ---
def lowercase(text): return text.lower()
def remove_html_tags(text): return re.sub(r'<.*?>', '', text)
def remove_urls(text): return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)
def remove_punctuation(text): return text.translate(str.maketrans('','', string.punctuation))
def chat_conversion(text):
    new_text = []
    for w in text.split():
        new_text.append(chat_words.get(w.upper(), w)) # Use .get() with default for words not in dict
    return " ".join(new_text)
def stopwords_removal(text):
    new_text = []
    for word in text.split():
        if word not in STOP_WORDS:
            new_text.append(word)
    return " ".join(new_text)
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags
        u"\U00002702-\U000027B0"  # Misc symbols
        u"\U000024C2-\U0001F251"  # Enclosed characters and others
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def expand_contractions(text): return contractions.fix(text)
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])
def tokenize(text): return word_tokenize(text) # Returns a list of tokens

def preprocess_text_pipeline(text):
    """
    Applies the full preprocessing pipeline to a single text string.
    Returns the cleaned text as a list of tokens.
    """
    text = lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = chat_conversion(text)
    text = stopwords_removal(text)
    text = remove_emoji(text)
    text = expand_contractions(text)
    text = lemmatize_text(text)
    tokens = tokenize(text) # This returns a list of tokens
    return tokens

# --- Main Preprocessing & Validation Function ---
def validate_and_preprocess_data_ge(raw_data_path, processed_data_output_dir, tokenizer_output_path):
    """
    Loads raw data, performs Great Expectations validation, preprocesses,
    splits data, tokenizes, pads sequences, and saves processed data and tokenizer.

    Args:
        raw_data_path (str): Path to the raw CSV file.
        processed_data_output_dir (str): Directory to save processed .npy files.
        tokenizer_output_path (str): Path to save the tokenizer.pkl file.
    """
    logger.info(f"Loading raw data from: {raw_data_path}")

    # Load data with Polars first
    try:
        df_polars_raw = pl.read_csv(raw_data_path)
        logger.info(f"Raw dataset loaded successfully from: {raw_data_path}")
    except FileNotFoundError:
        logger.error(f"Error: Raw data file '{raw_data_path}' not found.")
        sys.exit(1)

    # Perform initial dataset info and duplicate removal using Polars
    df_polars_raw = dataset_info(df_polars_raw)

    # Convert Polars DataFrame to Pandas DataFrame for Great Expectations (GE works best with Pandas)
    df_raw = df_polars_raw.to_pandas()

    # --- Great Expectations Validation (Raw Data) ---
    logger.info("Starting Great Expectations validation for raw data...")
    # Initialize an in-memory Data Context.
    context = gx.get_context()

    
    # Get the data source if it exists, otherwise add it.
    data_source_name = "my_pandas_datasource" 
    source_folder = (processed_data_output_dir)
    try:
        data_source = context.data_sources.get(data_source_name)
        logger.info(f"Attempting to use existing data_source named: {data_source_name} if it's existing")
    except (KeyError, gx.exceptions.DataContextError):
        logger.info(f"No data source named {data_source_name}. Creating new data source named: {data_source_name}")
        data_source = context.data_sources.add_pandas_filesystem(
            name=data_source_name,
            base_directory=source_folder # setting the home base (base directory)
        )

    # Create data asset
    data_asset_name = "raw_sentiment_data_asset_for_this_run"
    data_asset = data_source.add_csv_asset(name=data_asset_name)
    
    # create a batch definition
    # step1: retrieve data asset
    file_data_asset = context.data_sources.get(data_source_name).get_asset(data_asset_name)
    # step2: Add a Batch Definition to the Data Asset.
    batch_definition_name = raw_data_path.split("/")[-1]
    batch_definition_path = os.path.basename(raw_data_path)
    batch_definition = file_data_asset.add_batch_definition_path(
        name=batch_definition_name, path=batch_definition_path
    )
    # Verify the Batch Definition is valid.
    batch = batch_definition.get_batch()
    print(batch.head())

    # Define an Expectation Suite
    
    suite_name_raw = "raw_sentiment_data_expectation_suite"
    try:
         # Try to load existing suite using suites.get()
        suite_raw = context.suites.get(name=suite_name_raw)
        logger.info(f"Using existing raw data expectation suite: {suite_name_raw}")
    except gx.exceptions.DataContextError: # Use gx.exceptions
        # If it doesn't exist, create a new one
        logger.info(f"Creating new raw data expectation suite: {suite_name_raw}")

        # Create an empty suite object
        suite_raw = gx.ExpectationSuite(name=suite_name_raw) # <-- Create the suite object
        suite_raw = context.suites.add(suite_raw)
        suite_raw.add_expectation(gx.expectations.ExpectColumnToExist(column="review"))
        suite_raw.add_expectation(gx.expectations.ExpectColumnToExist(column="sentiment"))
        suite_raw.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="review"))
        suite_raw.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="sentiment"))
        suite_raw.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="sentiment", value_set=["positive", "negative"]))
        suite_raw.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(column="review", min_value=10, max_value=10000))
        logger.info(f"New expectation suite '{suite_name_raw}' created and expectations added.")
        


    # print("Here")
    # # --- GE Validation Execution (Updated for GE 1.3.6+ RuntimeBatchRequest) ---
    raw_validation_definition_name = "raw_data_validation_definition"
    try:
         # Try to get existing validation definition using validation_definitions.get()
        raw_validation_definition = context.validation_definitions.get(name=raw_validation_definition_name)
        logger.info(f"Using existing raw data Validation Definition: {raw_validation_definition_name}")
    except gx.exceptions.DataContextError:
        # If it doesn't exist, create a new one linking the batch template and the suite
        logger.info(f"Creating new raw data Validation Definition: {raw_validation_definition_name}")
        data_source_name = data_source_name
        data_asset_name = data_asset_name
        batch_definition = (
            context.data_sources.get(data_source_name)
            .get_asset(data_asset_name)
            .get_batch_definition(batch_definition_name)
        )

        raw_validation_definition = gx.ValidationDefinition( # <-- Create ValidationDefinition
            data=batch_definition, 
            suite=suite_raw,                    
            name=raw_validation_definition_name
        )
        context.validation_definitions.add(raw_validation_definition) # <-- Add to context

    try:
        logger.info(f"Running raw data Validation Definition '{raw_validation_definition_name}'...")

        # FIX: Instead of passing a 'dataframe', we pass a 'path'.
        # The path must match the identifier used in your BatchDefinition.
        path_to_validate = os.path.basename(raw_data_path)
        print(path_to_validate)
        validation_result_raw = raw_validation_definition.run(
            batch_parameters={"path": path_to_validate}
        )
        logger.info("Raw data validation complete.")

    except Exception as e:
        logger.error(f"Error during raw data GE Validation Definition run: {e}", exc_info=True)
        sys.exit(1)
    if not validation_result_raw.success:
        logger.error("\n!!! Raw Data Validation Failed !!!")
        logger.error(validation_result_raw)
        sys.exit(1)

    logger.info("Raw Data Validation Successful!")

    # --- Preprocessing ---
    logger.info("Starting data preprocessing...")
    df_processed = df_raw.copy()
    df_processed['cleaned_review'] = df_processed['review'].apply(preprocess_text_pipeline)
    df_processed['review_len'] = df_processed['cleaned_review'].apply(len)
    logger.info("Data preprocessing completed. Example of cleaned reviews:")
    logger.info("\n" + df_processed[['review', 'cleaned_review']].head(2).to_string())

    # # --- Great Expectations Validation (Processed Data) ---
    logger.info("Starting Great Expectations validation for processed data...")

      # Define the path for the new CSV file and save it 
    processed_data_filename = "processed_dummy_data.csv"
    processed_data_path = os.path.join(processed_data_output_dir, processed_data_filename)
    df_processed.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to: {processed_data_path}")

    ### SET UP THE FILE-BASED ASSET AND BATCH DEFINITION ###
    # Create a new data asset for the processed file.
    processed_asset_name = "processed_sentiment_data_asset"
    try:
        # Check if asset already exists to avoid errors on re-runs within the same script instance
        processed_data_asset = data_source.get_asset(processed_asset_name)
        logger.info(f" Check if asset already exists and get it")
    except LookupError:
        processed_data_asset = data_source.add_csv_asset(name=processed_asset_name)
        logger.info(f"Create processed asset named {processed_asset_name}")

    # create a batch definition
    # step1: retrieve data asset
    processed_data_asset = context.data_sources.get(data_source_name).get_asset("processed_sentiment_data_asset")
    # step2: Add a Batch Definition to the Data Asset.
    processed_batch_def_name = "processed_batch_def" 
    # The path is the relative path (just the filename) to the new CSV.
    processed_batch_def_path = os.path.basename(processed_data_path) 
    batch_definition_processed = processed_data_asset.add_batch_definition_path(
        name=processed_batch_def_name, path=processed_batch_def_path
    )
    # Verify the Batch Definition is valid.
    batch = batch_definition_processed.get_batch()
    print(batch.head())

    suite_name_processed = "processed_sentiment_data_expectation_suite"
    try:
         # Try to load existing suite using suites.get()
        processed_suite = context.suites.get(name=suite_name_processed)
        logger.info(f"Using existing raw data expectation suite: {suite_name_processed}")
    except gx.exceptions.DataContextError: # Use gx.exceptions
        # If it doesn't exist, create a new one
        logger.info(f"Creating new raw data expectation suite: {suite_name_processed}")

        # Create an empty suite object
        processed_suite = gx.ExpectationSuite(name=suite_name_processed) #  Create the suite object
        processed_suite = context.suites.add(processed_suite)
        processed_suite.add_expectation(gx.expectations.ExpectColumnToExist(
            column="cleaned_review"
        ))
        processed_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
            column="cleaned_review"
        ))
        processed_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
            column="cleaned_review",
            type_="object"
        ))
        processed_suite.add_expectation(gx.expectations.ExpectColumnToExist(
            column="review_len"
        ))
        processed_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
            column="review_len"
        ))
        processed_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
            column="review_len", 
            min_value=1, 
            max_value=10000
        ))
        logger.info(f"New expectation suite '{suite_name_processed}' created and expectations added.")


    processed_validation_definition_name = "processed_data_validation_definition"
    try:
         # Try to get existing validation definition using validation_definitions.get()
        processed_validation_definition = context.validation_definitions.get(
            name=processed_validation_definition_name
        )
        logger.info(f"Using existing raw data Validation Definition: {processed_validation_definition_name}")
    except gx.exceptions.DataContextError:
        # If it doesn't exist, create a new one linking the batch template and the suite
        logger.info(f"Creating new raw data Validation Definition: {processed_validation_definition_name}")
        batch_def_to_validate  = (
            context.data_sources.get(data_source_name)
            .get_asset(processed_asset_name)
            .get_batch_definition(processed_batch_def_name)
        )

        processed_validation_definition = gx.ValidationDefinition( # <-- Create ValidationDefinition
            data=batch_def_to_validate, 
            suite=processed_suite,                    
            name=processed_validation_definition_name
        )
        context.validation_definitions.add(processed_validation_definition) # <-- Add to context
    
    try:
        logger.info(f"Running processed data Validation Definition '{processed_validation_definition_name}'...")
        # Run the validation by passing the relative path to the processed file.
        validation_result_processed = processed_validation_definition.run(
            batch_parameters={"path": processed_batch_def_path}
        )
        logger.info("Processed data validation complete.")
    except Exception as e:
        logger.error(f"Error during processed data GE Validation Definition run: {e}", exc_info=True)
        sys.exit(1)
        
    if not validation_result_processed.success:
        logger.error("\n!!! Processed Data Validation Failed !!!")
        logger.error(validation_result_processed)
        sys.exit(1)

    logger.info("Processed Data Validation Successful!")


    
  
    # # --- Tokenization and Padding ---
    logger.info("Starting tokenization and padding...")
    calculated_max_len = int(df_processed['review_len'].max()) if not df_processed.empty else 0
    MAX_LEN_CAP = 500
    MAX_LEN = min(calculated_max_len, MAX_LEN_CAP) if calculated_max_len > 0 else MAX_LEN_CAP
    logger.info(f"Calculated MAX_LEN (before cap): {calculated_max_len}")
    logger.info(f"Using MAX_LEN for padding: {MAX_LEN}")

    if df_processed.empty:
        logger.warning("Processed DataFrame is empty. Skipping tokenization, padding, and splitting.")
        X_train, y_train, X_val, y_val, X_test, y_test = (np.array([]) for _ in range(6))
        tokenizer = Tokenizer()
    else:
        logger.info(f"Setting up tokenizer and pre-processing X and y dataset")
        tokenizer = Tokenizer(num_words=None, oov_token="<unk>")
        tokenizer.fit_on_texts(df_processed['cleaned_review'].tolist())
        X = tokenizer.texts_to_sequences(df_processed['cleaned_review'].tolist())
        X = pad_sequences(X, maxlen=MAX_LEN, padding='post')
        y = np.array([1 if label == "positive" else 0 for label in df_processed['sentiment']])
        
        try:
             X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
             X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        except ValueError as e:
             logger.error(f"Could not split data due to small dataset size or distribution: {e}. Ensure your raw data has enough samples for stratification.", exc_info=True)
             sys.exit(1)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- Save Processed Data and Tokenizer ---
    os.makedirs(processed_data_output_dir, exist_ok=True)
    
    # Save the numpy arrays
    np.save(os.path.join(processed_data_output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_data_output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_data_output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(processed_data_output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_data_output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(processed_data_output_dir, 'y_val.npy'), y_val)
    logger.info(f"Processed data (X_*.npy, y_*.npy) saved to: {processed_data_output_dir}")

    # Save the tokenizer
    with open(tokenizer_output_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to: {tokenizer_output_path}")

    logger.info("Data preprocessing and preparation complete.")


if __name__ == '__main__':
    # This block is for local testing of this script.
    # In a full pipeline, main.py will call this function.
    project_root = os.path.join(os.path.dirname(__file__), '..') # Assumes src/data_preprocessing.py
    raw_data_path_local = os.path.join(project_root, 'data', 'IMDB-Dataset.csv') # Assuming this is your raw data file
    processed_data_output_dir_local = os.path.join(project_root, 'data') # Outputs go here
    tokenizer_output_path_local = os.path.join(project_root, 'data', 'tokenizer.pkl')

    # Create a dummy IMDB-Dataset.csv if it doesn't exist for local testing
    if not os.path.exists(raw_data_path_local):
        print(f"Creating dummy raw data at {raw_data_path_local}")
        dummy_data = {
            'review': [
                "This movie was absolutely fantastic! I loved every moment of it. Highly recommend.",
                "Terrible experience, utterly disappointed. Waste of time and money. :(",
                "It's an okay film, nothing special. Could be better, could be worse. meh",
                "A truly great and inspiring movie. Loved the acting and the plot. :)",
                "Worst movie ever. I'm so angry I watched it. #fail",
                "Another positive review that is quite long and describes the plot in detail, covering various aspects like acting, direction, and cinematography. The story was compelling, and the characters were well-developed. I would watch this again.",
                 "Another negative review with lots of complaints about pacing, plot holes, and poor dialogue. The actors seemed bored, and the special effects were terrible. I wasted two hours of my life on this film."
            ],
            'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative', 'positive', 'negative'] # Use positive/negative
        }
        pd.DataFrame(dummy_data).to_csv(raw_data_path_local, index=False)
        print("Dummy raw data created.")

    # With the fluent API, no pre-existing gx/ folder or great_expectations.yml is required to run.
    validate_and_preprocess_data_ge(raw_data_path_local, processed_data_output_dir_local, tokenizer_output_path_local)