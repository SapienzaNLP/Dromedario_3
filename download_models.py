"""
Download a HuggingFace model on CINECA Leonardo login node.

Usage:
    python download_models.py --model meta-llama/Llama-3.1-8B-Instruct --token hf_xxx --hf-home /path/to/cache
    python download_models.py --model mistralai/Mistral-7B-v0.1  # public models, no token needed
"""

import os
from pathlib import Path

import dotenv
from tap import Tap

dotenv.load_dotenv()
user = os.environ.get("USER")
scratch_root = os.environ.get("SCRATCH_ROOT", f"/leonardo_scratch/large/userexternal/{user}")


class Args(Tap):
    model: str
    token: str | None = os.environ.get("HF_TOKEN", None)
    hf_home: str = f"{scratch_root}/.hf_cache"
    ignore_patterns: list[str] = ["*.bin"]  # noqa: RUF012
    # download safetensors only, skip pytorch bin


def main():
    args = Args()
    args = args.parse_args()

    # Set HF_HOME before importing huggingface_hub
    os.environ["HF_HOME"] = args.hf_home
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME = {args.hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Install it with: pip install huggingface_hub")
        raise SystemExit(1)

    print(f"Downloading: {args.model}")
    if args.ignore_patterns:
        print(f"Ignoring patterns: {args.ignore_patterns}")

    path = snapshot_download(
        repo_id=args.model,
        token=args.token,
        ignore_patterns=args.ignore_patterns,
    )

    print("\nDownload complete!")
    print(f"  Path: {path}")
    print("\nTo use it in your code:")
    print(f'  os.environ["HF_HOME"] = "{args.hf_home}"')
    print(f'  resolve_model_path("{args.model}")')


if __name__ == "__main__":
    main()
