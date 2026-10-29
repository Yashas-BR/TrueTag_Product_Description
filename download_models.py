"""Download the exact model versions TrueTag uses into ./models (about 6.5 GB).

    python download_models.py

The models are too large for GitHub (single files of 3.1 GB and 3.5 GB), so the
repository pins each one to a Hugging Face commit in config.json instead. This
script fetches those exact revisions, so every copy of the project runs the same
weights. After it finishes, the app loads from ./models and needs no internet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "spiece*"]


def local_dir(name: str) -> Path:
    return ROOT / "models" / name.replace("/", "__")


def main() -> None:
    from huggingface_hub import snapshot_download

    for name in (CONFIG["extractor"], CONFIG["llm"]):
        revision = CONFIG["revisions"][name]
        target = local_dir(name)
        print(f"{name} @ {revision[:7]} -> {target}")
        snapshot_download(name, revision=revision, local_dir=target, allow_patterns=PATTERNS)
    print("Done. Start the site with run_local.bat (Windows) or uvicorn (see README).")


if __name__ == "__main__":
    sys.exit(main())
