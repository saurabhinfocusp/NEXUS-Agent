"""Download one sample (paired histology image + h5ad expression) from the
HEST-1k dataset (https://github.com/mahmoodlab/HEST).

HEST-1k is hosted as a gated Hugging Face dataset (`MahmoodLab/hest`), so you
must:
  1. Accept the dataset's terms at https://huggingface.co/datasets/MahmoodLab/hest
  2. Create an access token at https://huggingface.co/settings/tokens
  3. Export it: `export HUGGING_FACE_HUB_TOKEN=hf_xxx`

Sample IDs look like `TENX95`, `INT1`, `NCBI783` -- see the dataset's
`HEST_v1_1_0.csv` metadata sheet on Hugging Face for the full id list.

Usage:
    python scripts/download_hest_sample.py TENX95
    python scripts/download_hest_sample.py TENX95 --output-dir ./hest_data
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import login, snapshot_download

REPO_ID = "MahmoodLab/hest"


def download_sample(sample_id: str, output_dir: Path, token: str | None = None) -> Path:
    if token:
        login(token=token)

    pattern = f"*{sample_id}[_.]**"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(output_dir),
    )

    seg_dir = output_dir / "cellvit_seg"
    if seg_dir.exists():
        for zip_path in seg_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(seg_dir)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_id", help="HEST-1k sample id, e.g. TENX95")
    parser.add_argument("--output-dir", default="./hest_data", help="Directory to download into (default: ./hest_data)")
    parser.add_argument(
        "--token",
        default=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face access token (default: $HUGGING_FACE_HUB_TOKEN)",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "no Hugging Face token found -- set HUGGING_FACE_HUB_TOKEN or pass --token "
            "(see https://huggingface.co/settings/tokens, and accept the dataset terms at "
            "https://huggingface.co/datasets/MahmoodLab/hest first)"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = download_sample(args.sample_id, output_dir, args.token)
    downloaded = sorted(p.relative_to(result_dir) for p in result_dir.rglob("*") if p.is_file())
    if not downloaded:
        print(f"No files matched sample id '{args.sample_id}' -- check the id against HEST_v1_1_0.csv on Hugging Face.")
        return

    print(f"Downloaded {len(downloaded)} file(s) to {result_dir}:")
    for path in downloaded:
        print(f"  {path}")


if __name__ == "__main__":
    main()
