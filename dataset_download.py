import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable HF transfer backend

import argparse
from huggingface_hub import HfApi, logging
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    subsets = ["BLIP3o/BLIP3o-Pretrain-Long-Caption", "BLIP3o/BLIP3o-Pretrain-Short-Caption"]

    api = HfApi()
    logging.set_verbosity_error()

    for subset in subsets:
        file_list = api.list_repo_files(subset, repo_type="dataset")

        info = api.dataset_info(subset, files_metadata=True)
        total_bytes = sum(sibling.size or 0 for sibling in info.siblings) / (1024 ** 3)
        print(f"Dataset size: {total_bytes:.2f} GB")

        output_dir = os.path.join(args.output_dir, subset.split("/")[-1])
        os.makedirs(output_dir, exist_ok=True)
        for file in tqdm(file_list):
            if not os.path.exists(os.path.join(output_dir, file)):
                api.hf_hub_download(repo_id=subset, filename=file, local_dir=output_dir, repo_type="dataset")
