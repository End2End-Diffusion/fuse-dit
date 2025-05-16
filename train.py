import argparse
import os
from omegaconf import OmegaConf

from diffusion.trainers import get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1) # only used for DeepSpeed
    parser.add_argument("--deepspeed_config_path", type=str, default=None) # only used for DeepSpeed
    args = parser.parse_args()

    # Handle both cases:
    # 1. DeepSpeed passes --local_rank as CLI argument
    # 2. torchrun sets LOCAL_RANK environment variable
    if args.local_rank == -1:
        # No CLI arg provided, check environment (torchrun case)
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    hparams = OmegaConf.load(args.config)
    trainer = get_trainer(hparams, args.local_rank, args.deepspeed_config_path)
    trainer.train()