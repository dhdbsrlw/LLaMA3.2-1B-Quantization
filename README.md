# LLaMA3.2-1B-Quantization Project
> In this project, we apply **Post-training Quantization (PTQ)** and **Layer Pruning** to the base model, and then fine-tune it using **LoRA**.

### 1. Donwload base model checkpoint from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B).
### 2. Create Conda environment.

```bash
conda create -n edge python=3.10 -y
conda activate edge
pip install -r requirements.txt
```
⚠️ Conda environment for AWQ
AutoAWQ requires a downgraded version of `transformers`.
To use AWQ, install dependencies with:
```bash
pip install -r requirements_awq.txt
```

<details>
<summary>Issues</summary>

  1. If an error related to the `pcre` package occurs, replace it with `re`.

  ```bash
  # /root/anaconda3/envs/edge/lib/python3.10/site-packages/gptqmodel/models/writer.py 
  import re # not pcre
  ```

  2. If an error occurs when using the `awq` package with some latest models (e.g., Qwen3), comment out the corresponding model entries.
  ```bash
  # /root/anaconda3/envs/edge_2/lib/python3.10/site-packages/awq/models/__init__.py
  # /root/anaconda3/envs/edge_2/lib/python3.10/site-packages/awq/models/auto.py 
  ```

</details>
  

### 3. Apply PTQ Methods.
#### 3.1 GPTQ
#### 3.2 AWQ


Block Sensitivity Analysis 
```bash
python src/block_analyze.py --config cfg/block_analyze.yaml
```

### Step 2. Prune Blocks and Quantize Model 
```bash
python src/block_prune.py --config cfg/block_prune.yaml
```

### Step 3. Prune Blocks and Quantize Model 
```bash
python src/block_prune.py --config cfg/block_prune.yaml
```

## Acknowledgement
This repository was built on the [ShortenedLLaMA](https://github.com/Nota-NetsPresso/shortened-llm) codebase.
