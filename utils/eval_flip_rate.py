import argparse
from pathlib import Path
import re
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


def collect_pairs(input_dir: Path) -> List[Tuple[Path, Path, int]]:
    """
    Find (images_XXXXX.png, inpaint_XXXXX.png) pairs and return with their index.
    """
    pairs: List[Tuple[Path, Path, int]] = []
    inpaint_files = sorted(input_dir.glob("inpaint_*.png"))
    pat = re.compile(r"inpaint_(\d+)\.png$")
    for ip in inpaint_files:
        m = pat.search(ip.name)
        if not m:
            continue
        idx = int(m.group(1))
        img = input_dir / f"images_{idx:05d}.png"
        if img.is_file():
            pairs.append((img, ip, idx))
    return pairs


@torch.no_grad()
def predict_top1(model, preprocess, device, images: List[Path]) -> torch.Tensor:
    """
    Predict top-1 class indices for a list of image paths.
    """
    batch = []
    for p in images:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
    x = torch.stack(batch, dim=0).to(device)
    logits = model(x)
    preds = logits.argmax(dim=1)
    return preds.cpu()


def main():
    ap = argparse.ArgumentParser(description="Compute flip rate to a given target class for flattened outputs.")
    ap.add_argument("--input_dir", required=True, help="Directory with images_XXXXX.png and inpaint_XXXXX.png")
    ap.add_argument("--target_id", required=True, type=int, help="Target ImageNet class id to consider a flip")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for classifier inference")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if not set)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    assert input_dir.is_dir(), f"Not a directory: {input_dir}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load classifier (ImageNet pretrained)
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=weights).to(device).eval()

    pairs = collect_pairs(input_dir)
    if len(pairs) == 0:
        print("No pairs found. Expect files like images_00001.png and inpaint_00001.png.")
        return

    total = 0
    target_hits = 0
    flips_to_target = 0
    skipped = 0

    # Process in batches
    for i in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[i : i + args.batch_size]
        orig_paths = [p[0] for p in batch_pairs]
        cf_paths = [p[1] for p in batch_pairs]

        try:
            orig_preds = predict_top1(model, preprocess, device, orig_paths)
            cf_preds = predict_top1(model, preprocess, device, cf_paths)
        except Exception as e:
            # If an image is corrupted, skip this batch
            print(f"Warning: skipping batch {i}-{i+len(batch_pairs)} due to error: {e}")
            skipped += len(batch_pairs)
            continue

        for op, cp in zip(orig_preds.tolist(), cf_preds.tolist()):
            total += 1
            if cp == args.target_id:
                target_hits += 1
                if op != args.target_id:
                    flips_to_target += 1

    print(f"Total pairs evaluated: {total} (skipped: {skipped})")
    if total == 0:
        return
    print(f"Target hits (CF == {args.target_id}): {target_hits} ({target_hits/total:.4f})")
    print(f"Flips to target (orig != {args.target_id} and CF == {args.target_id}): {flips_to_target} ({flips_to_target/total:.4f})")


if __name__ == "__main__":
    main()


