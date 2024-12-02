## Pico Dataset 

📔 A repository storing all of the script used to generate the Pico datasets (both for training and inference). The data used is just a re-processed version of the OLMO dataset. 

The datasets are hosted on the [Pico huggingface](https://huggingface.co/datasets/pico-lm/pretokenized-dolma) repo. 

The setup is fairly straight forward: 

Step 1. Create a `.env` file with the right `HF_TOKEN=...` 
Step 2. Run `download_data.sh`
Step 3. Run `create_idx_prepared_dataset.py`; to do this in a slurm environment just use the `create_prepared_dataset.sbatch` script. 
Step 4. Wait forever or buy a million CPUs. 

(Optional Step). Run throughout the `create_eval_batch.ipynb` to create the inference eval batch.