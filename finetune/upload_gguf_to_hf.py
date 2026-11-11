#!/usr/bin/env python3
"""Upload the VitalWatch GGUF builds + model card to the private HF repo.

Run from the finetune folder:
    python upload_gguf_to_hf.py

Auth: uses a cached huggingface token if present; otherwise prompts once -
paste your Write token from https://huggingface.co/settings/tokens.
"""

import os
import sys

REPO_ID = "TheBlindCoder/vitalwatch-slm-qwen-1p5b-gguf"

FILES = [
    ("hf_gguf_model_card.md", "README.md"),
    ("vitalwatch-slm-q8_0.gguf", "vitalwatch-slm-q8_0.gguf"),
    ("vitalwatch-slm-f16.gguf", "vitalwatch-slm-f16.gguf"),
]


def main():
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        sys.exit("huggingface_hub is not installed in THIS python.\n"
                 "Run:  python -m pip install -U \"huggingface_hub[cli]\"")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        login()  # interactive: paste your Write token

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    print("repo ready:", REPO_ID, "(private)")

    here = os.path.dirname(os.path.abspath(__file__))
    for local_name, remote_name in FILES:
        local_path = os.path.join(here, local_name)
        if not os.path.isfile(local_path):
            print(f"SKIP  {local_name} (not found in this folder)")
            continue
        size_gb = os.path.getsize(local_path) / 1e9
        print(f"uploading {local_name} ({size_gb:.2f} GB) -> {remote_name} ...")
        api.upload_file(path_or_fileobj=local_path, path_in_repo=remote_name,
                        repo_id=REPO_ID, repo_type="model")
        print("  done")

    print("\nall uploads complete: https://huggingface.co/" + REPO_ID)


if __name__ == "__main__":
    main()
