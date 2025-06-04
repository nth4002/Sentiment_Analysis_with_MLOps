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
import great_expectations as ge
from great_expectations.core import ExpectationConfiguration
from great_expectations.core import ExpectationSuite
import logging

# SET UP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # log to console
        logging.FileHandler("preprocessing.log") # log to file
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
STOP_WORDS = set(stopwords.words('english'))

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
    # Initialize Great Expectations context (non-interactive for scripting)
    context = ge.data_context.DataContext()
    
    batch_request_raw = {
        "runtime_parameters": {"batch_data": df_raw},
        "batch_identifiers": {"default_identifier_name": "raw_sentiment_data_batch"},
        "data_asset_name": "raw_sentiment_data_asset"
    }
    
    suite_name_raw = "raw_sentiment_data_expectation_suite"
    try:
        suite_raw = context.get_expectation_suite(suite_name_raw)
        logger.info(f"Using existing raw data expectation suite: {suite_name_raw}")
    except ge.exceptions.DataContextError:
        logger.info(f"Creating new raw data expectation suite: {suite_name_raw}")
        suite_raw = context.create_expectation_suite(suite_name_raw)
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_to_exist", kwargs={"column": "review"}
        ))
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_to_exist", kwargs={"column": "sentiment"}
        ))
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "review"}
        ))
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "sentiment"}
        ))
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set", kwargs={"column": "sentiment", "value_set": ["positive", "negative"]}
        ))
        suite_raw.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_value_lengths_to_be_between", kwargs={"column": "review", "min_value": 10, "max_value": 5000} # Adjust max based on your data
        ))
        context.save_expectation_suite(suite_raw)

    validator_raw = context.get_validator(
        batch_request=batch_request_raw,
        expectation_suite_name=suite_name_raw
    )
    validation_result_raw = validator_raw.validate()

    if not validation_result_raw.success:
        logger.error("\n!!! Raw Data Validation Failed !!!")
        logger.error(validation_result_raw.to_json_dict())
        # context.open_data_docs() # This opens a browser, not suitable for CI/CD
        sys.exit(1) # Stop the pipeline if validation fails

    logger.info("Raw Data Validation Successful!")

    # --- Preprocessing ---
    logger.info("Starting data preprocessing...")
    df_processed = df_raw.copy()
    df_processed['cleaned_review'] = df_processed['review'].apply(preprocess_text_pipeline)
    
    # Store length of processed reviews (list of tokens)
    df_processed['review_len'] = df_processed['cleaned_review'].apply(len)
    
    logger.info("Data preprocessing completed. Example of cleaned reviews:")
    logger.info(df_processed[['review', 'cleaned_review']].head(2))

    # --- Great Expectations Validation (Processed Data) ---
    logger.info("Starting Great Expectations validation for processed data...")
    batch_request_processed = {
        "runtime_parameters": {"batch_data": df_processed},
        "batch_identifiers": {"default_identifier_name": "processed_sentiment_data_batch"},
        "data_asset_name": "processed_sentiment_data_asset"
    }

    suite_name_processed = "processed_sentiment_data_expectation_suite"
    try:
        suite_processed = context.get_expectation_suite(suite_name_processed)
        logger.info(f"Using existing processed data expectation suite: {suite_name_processed}")
    except ge.exceptions.DataContextError:
        logger.info(f"Creating new processed data expectation suite: {suite_processed}")
        suite_processed = context.create_expectation_suite(suite_processed)
        suite_processed.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_to_exist", kwargs={"column": "cleaned_review"}
        ))
        suite_processed.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "cleaned_review"}
        ))
        suite_processed.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type", kwargs={"column": "cleaned_review", "type_": "list"}
        ))
        suite_processed.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "review_len"}
        ))
        suite_processed.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between", kwargs={"column": "review_len", "min_value": 1, "max_value": 1000} # Tokenized length, adjust as needed
        ))
        context.save_expectation_suite(suite_processed)

    validator_processed = context.get_validator(
        batch_request=batch_request_processed,
        expectation_suite_name=suite_name_processed
    )
    validation_result_processed = validator_processed.validate()

    if not validation_result_processed.success:
        logger.error("\n!!! Processed Data Validation Failed !!!")
        logger.error(validation_result_processed.to_json_dict())
        sys.exit(1) # Stop the pipeline

    logger.info("Processed Data Validation Successful!")

    # --- Tokenization and Padding ---
    logger.info("Starting tokenization and padding...")
    
    # max_len: find maximum length of review from data for proper padding
    MAX_LEN = int(df_processed['review_len'].max())
    logger.info(f"Calculated MAX_LEN for padding: {MAX_LEN}")

    # Initialize tokenizer with the preprocessed words (list of lists)
    tokenizer = Tokenizer(num_words=None, oov_token="<unk>") # num_words=None to keep all words, add OOV token
    tokenizer.fit_on_texts(df_processed['cleaned_review'])

    # Convert text to sequences of integers
    X = tokenizer.texts_to_sequences(df_processed['cleaned_review'])
    # Pad sequences to ensure uniform length
    X = pad_sequences(X, maxlen=MAX_LEN, padding='post') # Pad after the sequence

    # Convert sentiment labels to numeric (already done in source but ensure consistency)
    y = np.array([1 if label == "positive" else 0 for label in df_processed['sentiment']])

    # Split data into training, validation, and test sets
    logger.info("Splitting data into training, validation, and test sets (70/15/15 split)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # 0.5 of 0.3 = 0.15

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- Save Processed Data and Tokenizer ---
    os.makedirs(processed_data_output_dir, exist_ok=True) # Ensure output directory exists

    np.save(os.path.join(processed_data_output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_data_output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_data_output_dir, 'X_val.npy'), X_val)

    np.save(os.path.join(processed_data_output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_data_output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(processed_data_output_dir, 'y_val.npy'), y_val)

    with open(tokenizer_output_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    logger.info(f"Processed data (X_*.npy, y_*.npy) saved to: {processed_data_output_dir}")
    logger.info(f"Tokenizer saved to: {tokenizer_output_path}")
    logger.info("Data preprocessing and preparation complete.")

if __name__ == '__main__':
    # This block is for local testing of this script.
    # In a full pipeline, main.py will call this function.
    project_root = os.path.join(os.path.dirname(__file__), '..') # Assumes src/data_preprocessing.py
    raw_data_path_local = os.path.join(project_root, 'data', 'dummy_data.csv') # Assuming this is your raw data file
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
                "Worst movie ever. I'm so angry I watched it. #fail"
            ],
            'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative'] # Use positive/negative
        }
        pd.DataFrame(dummy_data).to_csv(raw_data_path_local, index=False)
        print("Dummy raw data created.")

    validate_and_preprocess_data_ge(raw_data_path_local, processed_data_output_dir_local, tokenizer_output_path_local)