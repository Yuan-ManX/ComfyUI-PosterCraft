# ComfyUI-PosterCraft

ComfyUI-PosterCraft is now available in ComfyUI, [PosterCraft](https://github.com/Ephemeral182/PosterCraft) is a unified framework for high-quality aesthetic poster generation that excels in precise text rendering, seamless integration of abstract art, striking layouts, and stylistic harmony.



## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-PosterCraft.git
```

3. Install dependencies:
```
cd ComfyUI-PosterCraft

# Create conda environment
conda create -n postercraft python=3.11
conda activate postercraft

# Install dependencies
pip install -r requirements.txt
```


## Model

### 💾 Model Zoo

We provide the weights for our core models, fine-tuned at different stages of the PosterCraft pipeline.

<div align="center">
<table>
<tr>
<th>Model</th>
<th>Stage</th>
<th>Description</th>
<th>Download</th>
</tr>
<tr>
<td>🎯 <b>PosterCraft-v1_RL</b></td>
<td>Stage 3: Aesthetic-Text RL</td>
<td>Optimized via Aesthetic-Text Preference Optimization for higher-order aesthetic trade-offs.</td>
<td><a href="https://huggingface.co/PosterCraft/PosterCraft-v1_RL">🤗 HF</a></td>
</tr>
<tr>
<td>🔄 <b>PosterCraft-v1_Reflect</b></td>
<td>Stage 4: Vision-Language Feedback</td>
<td>Iteratively refined using vision-language feedback for further harmony and content accuracy.</td>
<td><a href="https://huggingface.co/PosterCraft/PosterCraft-v1_Reflect">🤗 HF</a></td>
</tr>
</table>
</div>

---
