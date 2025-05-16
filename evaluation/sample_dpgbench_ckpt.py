import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json

import torch
from tqdm import tqdm
from PIL import Image

from accelerate import PartialState
from lightning import seed_everything

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, AutoencoderKLQwenImage
from transformers import GemmaTokenizer, AutoModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusion.models import DiT, FuseDiT
from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP

# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def create_grid(images, grid_size=(2, 2)):
    """Create a grid from a list of PIL images"""
    assert len(images) == grid_size[0] * grid_size[1], f"Expected {grid_size[0] * grid_size[1]} images, got {len(images)}"
    
    # Get dimensions of single image
    width, height = images[0].size
    
    # Create new image for grid
    grid_width = width * grid_size[1]
    grid_height = height * grid_size[0]
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid_image.paste(img, (col * width, row * height))
    
    return grid_image


def sample(args, data_dict):
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.save_dir, exist_ok=True)

    distributed_state = PartialState()

    state_dict = torch.load(f"{args.checkpoint}/{args.tag}/mp_rank_00_model_states.pt", map_location="cpu")
    if not args.ema_weights:
        # state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint, tag=str(args.tag))
        # Because we have removed the optimizer states...
        # Also, we are using zero2 so it should be fine
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

    # Process prompts as dict format {promptname: prompt}
    with distributed_state.split_between_processes(list(data_dict.items())[:args.num_samples]) as prompt_items:
        for prompt_name, prompt_text in tqdm(prompt_items):
            # Generate exactly 4 images
            with torch.autocast("cuda"):
                call_args = {
                    "prompt": prompt_text,
                    "width": args.resolution,
                    "height": args.resolution,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "num_images_per_prompt": 4,  # Always generate 4 images for 2x2 grid
                }
                if "fuse-dit" in args.type:
                    call_args["use_cache"] = True
                images = pipeline(**call_args)[0]
            
            # Create 2x2 grid from 4 images
            grid_image = create_grid(images, grid_size=(2, 2))
            
            # Save grid with prompt name
            output_path = os.path.join(args.save_dir, f"{prompt_name}.png")
            grid_image.save(output_path)


def main(args):
    with open(args.prompts) as f:
        data_dict = json.load(f)
    
    # Ensure data_dict is in the correct format {promptname: prompt}
    if isinstance(data_dict, list):
        # If it's a list of strings, convert to dict with numbered keys
        data_dict = {f"prompt_{i:05d}": prompt for i, prompt in enumerate(data_dict)}
    elif not isinstance(data_dict, dict):
        raise ValueError(f"Expected prompts file to contain a dict or list, got {type(data_dict)}")

    sample(args, data_dict)


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
    parser.add_argument("--prompts", type=str, default="prompts/mjhq30k.json")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4, help="Fixed at 4 for 2x2 grid generation")
    parser.add_argument("--save_dir", type=str, default="samples/self-attention-w2048-1b-res256-vaeflux-mjhq30k_100000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    main(args)