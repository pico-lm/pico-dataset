## Setup Instructions

### Prerequisites
- Hugging Face account with API token
- Python environment with dependencies installed (see `pyproject.toml`)

### Steps

1. **Configure Environment**
   - Create a `.env` file in the root directory
   - Add your Hugging Face token: `HF_TOKEN=your_token_here`

2. **Download Data**
   ```bash
   ./download_data.sh
   ```

3. **Prepare Dataset**
   - For local execution:
     ```bash
     python create_dolma_dataset.py
     ```
   - For SLURM environments:
     ```bash
     # Replace N with a shard index (0-99)
     sbatch create_prepared_dataset.sbatch N
     ```
   > ⚠️ Note: The shard index must be an integer between 0 and 99, representing which portion of the dataset to process.

4. **Optional: Create Evaluation Batch**
   - Open and run `create_paloma_dataset.ipynb` to generate inference evaluation data

5. **Optional: Further shard Dataset**
   - If you'd like to shard the dataest into smaller shards (e.g. 1,000/10,000 shards), you can run the script in `shard_dataset.ipynb`
   > Note that this will create a second version of the dataset, what we did is we ran this script to generate 10,000 shards and then deleted the original dataset
