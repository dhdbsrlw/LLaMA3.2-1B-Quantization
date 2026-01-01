# LLaMA3.2-1B-Quantization Project
> In this project, we apply **Post-training Quantization (PTQ)** and **Block Pruning** to the base model, and then fine-tune it using **LoRA**.

### 1. Donwload base model (LLaMA3.2-1B) checkpoint from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-1B).
### 2. Prepare the dataset (TinyStories-v2) from [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories).
### 3. Create Conda environment

```bash
conda create -n edge python=3.10 -y
conda activate edge
pip install -r requirements.txt
```
⚠️ Conda environment for AWQ
AutoAWQ requires a downgraded version of `transformers`.<br/>To use AWQ, install dependencies with:
```bash
pip install -r requirements_awq.txt
```

<details>
<summary>Conda Env Issues</summary>

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
  

### 4. Apply PTQ
#### 4.1 GPTQ
```bash
python src/quantize_gptq.py 
```
#### 4.2 AWQ
```bash
python src/quantize_awq.py 
```

### 5. Apply (Transformer) Block Pruning
#### 5.1. Analyze block sensitivity (ppl).
```bash
python src/block_analyze.py --cfg_path cfg/block_analyze.yaml
```
#### 5.2. Prune blocks and quantize model.
```bash
python src/block_prune.py --cfg_path cfg/block_prune.yaml
```
### 6. LoRA Fine-tuning
```bash
python src/train.py --cfg_path cfg/train.yaml
```

### 7. Evaluate
You can **directly use** the trained model. Download it here:
- LLaMA + GPTQ [[HF Link]](https://huggingface.co/dhdbsrlw/Llama_3_2_1B_GPTQ_4bit)
- LLaMA + LoRA Training [[HF Link]](https://huggingface.co/dhdbsrlw/Llama_3_2_1B_TinyStories_LoRA_SFT)
- LLaMA + Pruning 3 Blocks + LoRA Training [[HF Link]](https://huggingface.co/dhdbsrlw/Llama_3_2_1B_Prune3_LoRA) (final)

```bash
# (1) perplexity (generated text quality)
python src/eval/eval_ppl.py --cfg_path cfg/eval/ppl.yaml
# (2) throughput / latency
python src/eval/eval_time_gpuutil.py --cfg_path cfg/eval/time_gpuutil.yaml
# (3) model size (compresssion ratio)
python src/eval/eval_compression.py
```

## Acknowledgement
This repository was built on the [ShortenedLLaMA](https://github.com/Nota-NetsPresso/shortened-llm) codebase.
