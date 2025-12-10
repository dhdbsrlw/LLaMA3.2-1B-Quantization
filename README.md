# edge-ai
(in class) LLM Quantization Project

```bash
conda create -n edge python=3.10 -y
conda activate edge
pip install -r requirements.txt
```

```bash
conda activate edge 
```

- /root/anaconda3/envs/edge/lib/python3.10/site-packages/gptqmodel/models/writer.py 
```bash
import re # not pcre
```

---
```bash
conda activate edge_2
```
- /root/anaconda3/envs/edge_2/lib/python3.10/site-packages/awq/models/__init__.py
- /root/anaconda3/envs/edge_2/lib/python3.10/site-packages/awq/models/auto.py 

---
### Step 1. Block Sensitivity Analysis 
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