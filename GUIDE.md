
## Environment Variables
Some environment variables are needed for the project. Please create a `.env` like:
```
HF_CACHE_DIR=hf_cache
RAY_LOG_DIR=tmp
TMPDIR=tmp
TEMP=tmp
TMP=tmp
CUDA_DEVICE_ORDER=PCI_BUS_ID
RAY_DEBUG=legacy
ROOT_DIR=$pwd
HF_TOKEN=[for huggingface connection]
WANDB_API_KEY=[for wandb connection]
```
Then run `export $(grep -v '^#' .env | xargs)` to export them. 

## Preparation
#### Step 1: Data 
Run `python hf_data_download.py` to download all the training and eval data into `verl_data/`, including:
- training data of different mixtures (`if{X}_logicif[Y].parquet`): X IF instructions and Y LogicIF instructions
- eval data (`ifbench_294_ifeval_541_logicif_749.parquet`): All the instructions from IFBench, IFEval, and LogicIFEval-mini

#### Step 2: Base Models
Run `python hf_model_download.py` to download the base models from huggingface. We only train Qwen3 models at this stage.



## Docker Setup

#### Step 1: Build the Docker Image

```bash
docker build -t rlif:latest .
```

#### Step 2: Run the Container
```bash
docker run --gpus all -it -v $(pwd):/workspace -w workspace rlif:latest
```

- The `-v $(pwd):/workspace` flag mounts your current directory into the container
- The `--gpus all` flag enables GPU access inside the container


## Training
#### Step 1: Generate Recipes and Training
Run `python generate_train_recipes.py` to generate training recipes for different runs. They are located in `recipe/rlif/`. For the training, the training set size is 80000 and we train it for 10 epochs with max steps 500 by default, with a learning rate of 1e-6 and glbal batch size of 512. The `max_prompt_length` is set to 2048. The experiments are expected to run with a node (no ray hanging issues). Then submit slurm job below to test the training on small model and data:
```
sbatch recipe/rlif/slurm_qwen06b_think_if1000_logicif1000_grpo.sh
```
If OOM is encountered, considering decrease the value of `ppo_micro_batch_size` to 8.



- The training log is saved to `log/`
- The checkpoints are saved to `checkpoints/`

#### Step 2: Convert FSDP checkpoints to Huggingface format
(Skip, I will do this on my end)

#### Step 3: Upload the Trained Models.
Run `python hf_upload.py checkpoints` to upload the trained models.

## Project Statges
#### Stage 1
For the first stage, we need to figure out the affects of two instruction following abilities, traditional constrains-based IF and our logical IF, on the other tasks. So I need the following models:
1. Qwen3-8B (think) trained on `if80000.parquet` 
2. Qwen3-8B (nothink) trained on `if80000.parquet` 
3. Qwen3-8B (think) trained on `logicif80000.parquet` 
4. Qwen3-8B (nothink) trained on `logicif80000.parquet` 

5. Qwen3-1.7B (think) trained on `if80000.parquet` 
6. Qwen3-1.7B (nothink) trained on `if80000.parquet` 
7. Qwen3-1.7B (think) trained on `logicif80000.parquet` 
8. Qwen3-1.7B (nothink) trained on `logicif80000.parquet` 

Please run `python generate_train_recipes.py` to generate the corresponding training scripts. After I get these models, I will test them on various benchmarks lile reasoning, tool calling, coding and so forth.
