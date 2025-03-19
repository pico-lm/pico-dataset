"""
Create the dolma-pretokenized dataset: 
A pretokenized, pre-shuffled dolma dataset with documents separated by boundary markers, and randomly shuffled
at both the document and example levels.

This script:
1. Loads a portion of the Dolma dataset from Hugging Face
2. Splits it into manageable shards
3. Tokenizes and chunks the text into fixed-length sequences
4. Uploads the processed data to a private Hugging Face repository

NOTE: Creates dataset in shards of 204800000 examples. With a total of 100 shards, you would need
to run this script 100 times to create a full dataset.

Usage:
    python create_dolma_dataset.py --idx <shard_index> --num_workers <worker_count>
"""

import os
import huggingface_hub as hf
from transformers import AutoTokenizer
from datasets import load_dataset
import click

from datasets.utils.logging import set_verbosity_debug
import logging

# Set up verbose logging to help with debugging
set_verbosity_debug()  # Most verbose
logging.getLogger("datasets").setLevel(logging.DEBUG)
logger = logging.getLogger("dolma_processing")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants
NUM_WORKERS = 14  # Number of parallel processes for data processing
NUM_SHARDS = 100  # Total number of data shards to create
TOTAL_EXAMPLES = 204800000  # Target total number of examples across all shards
                            # Note: Comment in code says 1000 shards but NUM_SHARDS is set to 100

HF_REPO = "pico-lm/pretokenized-dolma"  # HF repository to store the processed data
SEQ_LEN = 2048 + 1  # Sequence length (include +1 for right-shifting during training)

# Load Hugging Face Token from .env file for authentication
with open(".env", "r") as file:
    for line in file:
        if line.startswith("HF_TOKEN"):
            HF_TOKEN = line.split("=")[1].strip()
            break

# Initialize Hugging Face API and ensure the repository exists
api = hf.HfApi()
api.create_repo(HF_REPO, private=True, exist_ok=True, token=HF_TOKEN, repo_type="dataset")

# Load the tokenizer used for processing the text
# OLMo is an open language model from AI2
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")

def tokenize_and_chunk(examples):
    """
    Tokenizes and chunks text data to fixed-length sequences.
    
    Args:
        examples: A batch of text examples from the dataset
        
    Returns:
        Dictionary containing chunked token sequences of length SEQ_LEN
    """
    tokens = []
    # Process each text example in the batch
    for text in examples['text']:
        # Convert text to token IDs
        _tokens = tokenizer.encode(text)
        # Add EOS token to mark the end of each text example
        _tokens.append(tokenizer.eos_token_id)
        # Accumulate all tokens in a flat list
        tokens.extend(_tokens)

    # Split the accumulated tokens into chunks of SEQ_LEN
    chunks = [tokens[i:i + SEQ_LEN] for i in range(0, len(tokens), SEQ_LEN)]
    
    # Discard the last chunk if it's shorter than SEQ_LEN to ensure uniform sequence length
    if len(chunks[-1]) < SEQ_LEN:
        chunks = chunks[:-1]
        
    return {'input_ids': chunks}


@click.command()
@click.option("--idx", type=int, default=0, help="Index of the shard to process (0 to NUM_SHARDS-1)")
@click.option("--num_workers", type=int, default=NUM_WORKERS, help="Number of parallel workers for processing")
def create_prepared_dataset(idx: int, num_workers: int):
    """
    Main function to create and upload the prepared dataset.
    
    Args:
        idx: Index of the shard to process
        num_workers: Number of parallel workers to use for processing
    """
    # Load the Dolma dataset from Hugging Face
    dolma_dataset = load_dataset(
        "dolma", 
        split="train", 
        trust_remote_code=True,  # Required for custom processing code in the dataset
        cache_dir="data/hf_cache",  # Local cache to avoid repeated downloads
    )
    
    # Shuffle the dataset and select only 30% of it to reduce size
    dolma_dataset = dolma_dataset.shuffle(seed=420).select(range(int(len(dolma_dataset)*0.3)))

    # Extract the specific shard we're processing in this run
    dolma_dataset_subset = dolma_dataset.shard(
        num_shards=NUM_SHARDS, 
        index=idx, 
        contiguous=True,  # Ensures data is split into contiguous chunks
        # keep_in_memory=True,  # Commented out to avoid memory issues with large datasets
        writer_batch_size=100_000  # Controls memory usage during writing
    )

    print(f"Processing shard {idx} -- {len(dolma_dataset_subset)} examples")

    # Set up directory for storing processed shards locally
    shard_directory = "data/hf_cache/dolma/pretokenized"
    os.makedirs(shard_directory, exist_ok=True)

    # Define paths for local storage and HF hub
    # Format: train-000-of-100.parquet for shard 0 of 100
    shard_filename = f"train-{idx:03d}-of-{NUM_SHARDS:03d}.parquet"
    shard_path = os.path.join(shard_directory, shard_filename)
    hf_file_path = f"data/{shard_filename}"

    # Skip processing if shard already exists, just upload it
    if os.path.exists(shard_path):
        print(f"\t Shard already exists locally, uploading to Hugging Face Hub")
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=hf_file_path,
            repo_id=HF_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
        return

    # Tokenize and chunk the text data using parallel processing
    tokenized_dataset_shard = dolma_dataset_subset.map(
        tokenize_and_chunk,
        remove_columns=dolma_dataset_subset.column_names,  # Remove original columns, keep only tokens
        batched=True,  # Process data in batches for efficiency
        batch_size=350,  # Number of examples per batch
        num_proc=num_workers,  # Number of parallel processes
        keep_in_memory=True,  # Keep processed data in memory for faster processing
        new_fingerprint=f"fingerprint-train-{idx:03d}-of-{NUM_SHARDS:03d}",  # Cache fingerprint
    )

    # Shuffle the tokenized data to remove any remaining patterns
    print(f"\t Shuffling shard")
    tokenized_dataset_shard = tokenized_dataset_shard.shuffle(seed=42)

    # Sample a fixed number of examples from the shard to ensure all shards have same size
    print(f"\t Sampling shard")
    examples_per_shard = TOTAL_EXAMPLES // NUM_SHARDS
    tokenized_dataset_shard = tokenized_dataset_shard.take(examples_per_shard)

    # Save the processed shard to local disk as a Parquet file
    print(f"\t Saving shard to local disk")
    tokenized_dataset_shard.to_parquet(shard_path)

    # Upload the shard to Hugging Face Hub for storage and sharing
    print(f"\t Uploading shard to Hugging Face Hub")
    api.upload_file(
        path_or_fileobj=shard_path,
        path_in_repo=hf_file_path,
        repo_id=HF_REPO,
        token=HF_TOKEN,
        repo_type="dataset",
    )


if __name__ == "__main__":
    create_prepared_dataset()
