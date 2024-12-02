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
     python create_idx_prepared_dataset.py
     ```
   - For SLURM environments:
     ```bash
     sbatch create_prepared_dataset.sbatch
     ```
   > ⚠️ Note: This process is computationally intensive and may take considerable time.

4. **Optional: Create Evaluation Batch**
   - Open and run `create_eval_batch.ipynb` to generate inference evaluation data