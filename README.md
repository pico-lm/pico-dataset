# Pico Dataset
A toolkit for creating, and processing the [pretokenized-dolma](https://huggingface.co/datasets/pico-lm/pretokenized-dolma) and [pretokenized-paloma](https://huggingface.co/datasets/pico-lm/pretokenized-paloma) datasets available on our [HuggingFace org](https://huggingface.co/pico-lm).

This repository contains tools and scripts for creating datasets derived from the [Dolma](https://allenai.org/dolma) and [Paloma](https://allenai.org/evaluation-frameworks) corpora, suitable for training and evaluating large language models. The toolkit provides functionality for downloading source data, preprocessing, sharding, and preparing evaluation datasets.

> ⚠️ Note: To run these instructions you need A LOT of storage; roughly on the order of 20TB. 

The goal of releasing these scripts is for transparency, we encourage you to just use our uploaded datasets directly rather than reproducing the work.

### Prerequisites
- Hugging Face account with API token
- Python environment with dependencies installed (see `pyproject.toml`)

### Steps

1. **Configure Environment**

   - Create a `.env` file in the root directory and add your Hugging Face token: `HF_TOKEN=your_token_here`
   - Run `poetry install` to get dependencies installed
   - Run `poetry shell` to launch virtual env

2. **Download Data**

   Run the following to automatically download data from `https://huggingface.co/datasets/allenai/dolma` (downloads on the order of 10TB of data).

   ```bash
   ./download_data.sh
   ```

3. **Create Dolma Dataset**

   To create the `pretokenized-dolma` dataset run the following script: 

   ```bash
   python create_dolma_dataset.py --idx ... --num_workers ...
   ```

   Note that the argument following `--idx` is required, and specifies what shard of the dataset to process. It needs to be an integer between 0 and 99.
   
   The second argument `--num_workers` - the number of workers - is optional but ideally should be ~ the number of cpus available.

4. **Create Evaluation Batch**

   Open and run `create_paloma_dataset.ipynb` to generate the `pretokenized-paloma` and `pretokenized-paloma-tinsy` datasets.

5. **Optional: Further shard Dolma Dataset**

   The original `create_dolma_dataset.py` script shards the dataset into 100 shards. We realized in hindsight that it would be better to shard the dataset into smaller chunks for easier loading.
   You can open and run `finegrain_shard_dolma_dataset.ipynb` to further shard the preprocessed dolma dataset into 10,000 shards
   > Note that this will create a second version of the dataset, what we did is we ran this script to generate 10,000 shards and then deleted the original dataset
