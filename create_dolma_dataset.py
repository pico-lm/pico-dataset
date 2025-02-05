"""
Create a prepared dataset: a dataset from DOLMA that has been pre-tokenized, with documents separated 
by boundary markers, and randomly shuffled.

NOTE: Creates dataset in shards of 204800000 examples. With a total of 100 shards, you would need
to run this script 100 times to create a full dataset.
"""

import os
import huggingface_hub as hf
from transformers import AutoTokenizer
from datasets import load_dataset
import click

from datasets.utils.logging import set_verbosity_debug
import logging

set_verbosity_debug()  # Most verbose
logging.getLogger("datasets").setLevel(logging.DEBUG)
logger = logging.getLogger("dolma_processing")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants
NUM_WORKERS = 14
NUM_SHARDS = 100
TOTAL_EXAMPLES = 204800000  # Target total number of examples for 1000 shards

HF_REPO = "pico-lm/pretokenized-dolma"
SEQ_LEN = 2048 + 1  # Sequence length (include +1 for right-shifting)

# Load HF Token from .env file
with open(".env", "r") as file:
    for line in file:
        if line.startswith("HF_TOKEN"):
            HF_TOKEN = line.split("=")[1].strip()
            break

# Initialize Hugging Face API
api = hf.HfApi()
api.create_repo(HF_REPO, private=True, exist_ok=True, token=HF_TOKEN, repo_type="dataset")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")

def tokenize_and_chunk(examples):
    """Tokenizes and chunks text data to SEQ_LEN sequences."""
    tokens = []
    for text in examples['text']:
        _tokens = tokenizer.encode(text)
        _tokens.append(tokenizer.eos_token_id)
        tokens.extend(_tokens)

    # Split tokens into chunks of SEQ_LEN
    chunks = [tokens[i:i + SEQ_LEN] for i in range(0, len(tokens), SEQ_LEN)]
    
    # Discard the last chunk if it's shorter than SEQ_LEN
    if len(chunks[-1]) < SEQ_LEN:
        chunks = chunks[:-1]
        
    return {'input_ids': chunks}


@click.command()
@click.option("--idx", type=int, default=0)
@click.option("--num_workers", type=int, default=NUM_WORKERS)
def create_prepared_dataset(idx: int, num_workers: int):
    """Main function to create and upload the prepared dataset."""
    # Load and prepare the dataset
    dolma_dataset = load_dataset(
        "dolma", 
        split="train", 
        trust_remote_code=True, 
        cache_dir="data/hf_cache",
    )
    dolma_dataset = dolma_dataset.shuffle(seed=420).select(range(int(len(dolma_dataset)*0.3)))

    dolma_dataset_subset = dolma_dataset.shard(
        num_shards=NUM_SHARDS, 
        index=idx, 
        contiguous=True, 
        # keep_in_memory=True, 
        writer_batch_size=100_000
    )

    print(f"Processing shard {idx} -- {len(dolma_dataset_subset)} examples")

    shard_directory = "data/hf_cache/dolma/pretokenized"
    os.makedirs(shard_directory, exist_ok=True)

    # Define paths for local storage and HF hub
    shard_filename = f"train-{idx:03d}-of-{NUM_SHARDS:03d}.parquet"
    shard_path = os.path.join(shard_directory, shard_filename)
    hf_file_path = f"data/{shard_filename}"

    # Skip processing if shard already exists
    if os.path.exists(shard_path):
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=hf_file_path,
            repo_id=HF_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
        return

    # Tokenize, shuffle, and sample
    tokenized_dataset_shard = dolma_dataset_subset.map(
        tokenize_and_chunk,
        remove_columns=dolma_dataset_subset.column_names,
        batched=True,
        batch_size=350,
        num_proc=num_workers,
        keep_in_memory=True,
        new_fingerprint=f"fingerprint-train-{idx:03d}-of-{NUM_SHARDS:03d}",
    )

    print(f"\t Shuffling shard")
    tokenized_dataset_shard = tokenized_dataset_shard.shuffle(seed=42)

    print(f"\t Sampling shard")
    tokenized_dataset_shard = tokenized_dataset_shard.take(TOTAL_EXAMPLES // NUM_SHARDS)

    # Save shard locally
    print(f"\t Saving shard to local disk")
    tokenized_dataset_shard.to_parquet(shard_path)

    # Upload shard to Hugging Face Hub
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
