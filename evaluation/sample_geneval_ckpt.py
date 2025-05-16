"""
Adapted from:
https://github.com/djghosh13/geneval/blob/main/generation/diffusers_generate.py

Modified to use custom checkpoint loading while maintaining geneval save format
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json

import torch
from tqdm import tqdm

from accelerate import PartialState
from lightning import seed_everything

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, AutoencoderKLQwenImage
from transformers import GemmaTokenizer, AutoModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusion.models import DiT, FuseDiT
from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP


def load_pipeline(args):
    """Load pipeline from checkpoint using the original format"""
    state_dict = torch.load(f"{args.checkpoint}/{args.tag}/mp_rank_00_model_states.pt", map_location="cpu")
    if not args.ema_weights:
        state_dict = state_dict["module"]
    else:
        state_dict = state_dict["ema"]

    for k in list(state_dict.keys()):
        if "_orig_mod." in k:
            state_dict[k.replace("_orig_mod.", "")] = state_dict[k]
            del state_dict[k]

    if args.type == "baseline-dit":
        config = DiTConfig.from_pretrained(args.checkpoint)
        transformer = DiT(config)
        transformer.load_state_dict(state_dict)
    elif "fuse-dit" in args.type:
        config = FuseDiTConfig.from_pretrained(args.checkpoint)
        transformer = FuseDiT(config)
        transformer.load_state_dict(state_dict)

    tokenizer = GemmaTokenizer.from_pretrained(config.base_config._name_or_path)
    if "qwen" in args.vae.lower():
        vae = AutoencoderKLQwenImage.from_pretrained(args.vae, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae")

    if args.type == "baseline-dit":
        pipeline = DiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=vae,
            tokenizer=tokenizer,
            llm=AutoModel.from_pretrained(config.base_config._name_or_path),
        )
    elif args.type == "fuse-dit":
        pipeline = FuseDiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=vae,
            tokenizer=tokenizer
        )
    elif args.type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=vae,
            tokenizer=tokenizer,
            clip=CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder"),
            clip_tokenizer=CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer"),
        )
    else:
        raise ValueError(f"Unknown type: {args.type}")

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def generate(args):
    # Initialize seed and settings early
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize distributed state BEFORE loading pipeline
    distributed_state = PartialState()
    
    # Load metadata
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # Load pipeline AFTER distributed state initialization
    pipeline = load_pipeline(args)
    
    # Split work between processes
    with distributed_state.split_between_processes(list(enumerate(metadatas))) as samples:
        for index, metadata in tqdm(samples):
            outpath = os.path.join(args.save_dir, f"{index:0>5}")
            os.makedirs(outpath, exist_ok=True)

            prompt = metadata['prompt']
            batch_size = args.batch_size
            # print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)

            sample_count = 0
            for _ in range((args.n_samples + batch_size - 1) // batch_size):
                # Generate images
                with torch.autocast("cuda"):
                    call_args = {
                        "prompt": prompt,
                        "width": args.resolution,
                        "height": args.resolution,
                        "num_inference_steps": args.num_inference_steps,
                        "guidance_scale": args.guidance_scale,
                        "num_images_per_prompt": min(batch_size, args.n_samples - sample_count),
                    }
                    if "fuse-dit" in args.type:
                        call_args["use_cache"] = True
                    images = pipeline(**call_args)[0]
                for image in images:
                    image.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1


def main(args):
    generate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/sa-w2048-3b-res256-e2efluxvae-fulldataset-500k")
    parser.add_argument("--type", type=str, default="baseline-dit")
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--tag", type=str, default="100000")
    parser.add_argument("--vae", type=str, default="REPA-E/e2e-flux-vae")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_g", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--t5", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--metadata_file", type=str, default="prompts/geneval_metadata.jsonl", help="Path to metadata.jsonl file")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="samples/self-attention-w2048-1b-res256-vaeflux-geneval_100000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    main(args)