<h1 align="center">🚀 REPA-E <em>for</em> T2I</h1>

<p align="center">
  <em>End-to-End Tuned VAEs for Supercharging Text-to-Image Diffusion Transformers</em>
</p>

<p align="center">
  <a href="https://End2End-Diffusion.github.io/repa-e-t2i">🌐 Project Page</a> &ensp;
  <a href="https://huggingface.co/REPA-E/models">🤗 Models</a> &ensp;
  <a href="https://arxiv.org/abs/2504.10483">📃 Paper</a> &ensp;
  <br><br>
  <!-- <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=repa-e-unlocking-vae-for-end-to-end-tuning-of"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/repa-e-unlocking-vae-for-end-to-end-tuning-of/image-generation-on-imagenet-256x256" alt="PWC"></a> -->
</p>

## 📢 Overview

We present **REPA-E for T2I**, a family of End-to-End Tuned VAEs for supercharging text-to-image generation training. End-to-end VAEs show superior performance across all T2I benchmarks (COCO30k, DPG-Bench, GenAI-Bench, GenEval, MJHQ30k) with improved semantic spatial structure and details, without need for any additional representation alignment losses.

This repository is a fork of [Fuse-DiT](https://github.com/tang-bd/fuse-dit) adapted for training **REPA-E models**.

### Key Highlights

- **🏆 Superior T2I Performance**: Consistently outperform standard VAEs across all T2I benchmarks (COCO-30K, DPG-Bench, GenAI-Bench, GenEval, MJHQ-30K) without additional alignment losses
- **🎯 ImageNet → T2I Generalization**: End-to-end tuning on ImageNet 256×256 generalizes to T2I generation across multiple resolutions (256×256, 512×512)
- **🥇 SOTA on ImageNet**: Achieve state-of-the-art gFID 1.12 on ImageNet 256×256 with classifier-free guidance
- **⚡ Faster Convergence**: Enable quicker training of diffusion models with improved semantic spatial structure and latent representations

## 🎁 Model Release

All REPA-E VAE models are publicly available and ready to use in your text-to-image pipelines.

### 📦 Available Models

Pre-trained end-to-end tuned VAEs are available on Hugging Face:

| Model | Hugging Face Link |
|-------|-------------------|
| **E2E-FLUX-VAE** | 🤗 [REPA-E/e2e-flux-vae](https://huggingface.co/REPA-E/e2e-flux-vae) |
| **E2E-SD-3.5-VAE** | 🤗 [REPA-E/e2e-sd3.5-vae](https://huggingface.co/REPA-E/e2e-sd3.5-vae) |
| **E2E-Qwen-Image-VAE** | 🤗 [REPA-E/e2e-qwenimage-vae](https://huggingface.co/REPA-E/e2e-qwenimage-vae) |

### ⚡ Quickstart

Using REPA-E VAEs is as simple as loading from Hugging Face:

```python
from diffusers import AutoencoderKL, AutoencoderKLQwenImage

# Load end-to-end tuned FLUX / Qwen-Image VAE
vae = AutoencoderKL.from_pretrained("REPA-E/e2e-flux-vae").to("cuda")
vae = AutoencoderKLQwenImage.from_pretrained("REPA-E/e2e-qwenimage-vae").to("cuda")

# Use in your pipeline with vae.encode(...) / vae.decode(...)
```

For complete usage examples and integration with diffusion models, please see the [Hugging Face Models](https://huggingface.co/REPA-E/models).

---

## 🔬 Training Your Own Models

This repository allows you to train diffusion transformers with different VAEs to compare the training speedup and performance gains of end-to-end tuned VAEs versus standard ones. Below is how you can set up the environment and start training.

## 🛠️ Setup

Create a virtual environment (Python~=3.10):

```bash
conda create -n fuse-dit python=3.10
conda activate fuse-dit
```

Clone the repository:

```bash
git clone https://github.com/End2End-Diffusion/fuse-dit.git
cd fuse-dit
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

To download and prepare the training datasets, run:

```bash
python dataset_download.py
```

This script downloads the datasets into the `data/` directory with the following structure:
```
data/
├── BLIP3o-Pretrain-Long-Caption/
└── BLIP3o-Pretrain-Short-Caption/
```

### Configuring Dataset Paths

After downloading, update the dataset paths in all config files under `configs/e2e-vae/`. Replace the placeholder `/PATH/TO/DATASET/` with your actual dataset directory path (e.g., `/home/user/fuse-dit/data/`).

## 🚂 Training

To launch training on GPU devices with SLURM:

```bash
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=7234

srun torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    -c configs/e2e-vae/<config_file>.yaml \
    --deepspeed_config_path configs/deepspeed/zero2.json
```

### Available Configurations

The [configs/e2e-vae](configs/e2e-vae) folder contains 6 configurations organized in 3 pairs. Each pair compares training convergence between the original VAE and the end-to-end tuned VAE:

**FLUX VAE:**
- `sa-w2048-3b-res256-fluxvae-fulldataset-500k.yaml` - Original FLUX VAE
- `sa-w2048-3b-res256-e2efluxvae-fulldataset-500k.yaml` - End-to-end tuned FLUX VAE

**Stable Diffusion 3.5 VAE:**
- `sa-w2048-3b-res256-sd3.5vae-fulldataset-500k.yaml` - Original SD-3.5 VAE
- `sa-w2048-3b-res256-e2esd3.5vae-fulldataset-500k.yaml` - End-to-end tuned SD-3.5 VAE

**Qwen-Image VAE:**
- `sa-w2048-3b-res256-qwenimagevae-fulldataset-500k.yaml` - Original Qwen-Image VAE
- `sa-w2048-3b-res256-e2eqwenimagevae-fulldataset-500k.yaml` - End-to-end tuned Qwen-Image VAE

## 📈 Evaluation

Generate samples from trained checkpoints using the evaluation scripts. All scripts follow the same interface and support distributed generation via `accelerate`.

### Sample Generation

```bash
accelerate launch --num_processes=8 --num_machines=1 evaluation/sample_<benchmark>_ckpt.py \
    --checkpoint <checkpoint_dir> \
    --tag <iteration_step> \
    --type baseline-dit \
    --vae <vae_path> \
    --prompts prompts/<benchmark>.json \
    --resolution 256 \
    --num_inference_steps 25 \
    --guidance_scale 6.0 \
    --batch_size 64 \
    --save_dir <output_dir> \
    --ema_weights
```

**Parameters:**
- `--checkpoint`: Training checkpoint directory (e.g., `output/sa-w2048-3b-res256-e2efluxvae-fulldataset-500k`)
- `--tag`: Training iteration checkpoint to load (e.g., `100000`, `200000`)
- `--type`: Model architecture, use `baseline-dit`
- `--vae`: VAE model path (original or end-to-end tuned variant)
- `--prompts`: Prompt file from `prompts/` directory (e.g., `prompts/coco30k.json`, `prompts/mjhq30k.json`, `prompts/geneval_metadata.jsonl`)
- `--resolution`: Generation resolution in pixels
- `--num_inference_steps`: Number of denoising steps
- `--guidance_scale`: Classifier-free guidance strength
- `--batch_size`: Samples per forward pass per GPU
- `--save_dir`: Output directory for generated images
- `--ema_weights`: Load EMA weights instead of training weights (enabled by default)

**Available benchmarks:** `coco`, `mjhq`, `geneval`, `dpgbench`, `genaibench`

For metric computation, refer to the respective benchmark evaluation protocols. For FID scores, it can be computed using `evaluation/fid.py`. Other metrics can be computed using the official benchmark repositories.

## 📝 Citation

If you find our work useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{leng2025repae,
  title={REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers},
  author={Xingjian Leng and Jaskirat Singh and Yunzhong Hou and Zhenchang Xing and Saining Xie and Liang Zheng},
  year={2025},
  journal={arXiv preprint arXiv:2504.10483},
}
```

## 🙏 Acknowledgements

We thank the [Fuse-DiT](https://github.com/tang-bd/fuse-dit) team for open-sourcing their training codebase.
