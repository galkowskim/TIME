#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import List


def find_cf_images(results_dir: Path) -> List[Path]:
    """
    Find all counterfactual images under:
      Results/<exp_name>/*/*/CF/*.png
    """
    cf_dirs = []
    # Results/<exp_name> exists; dive into all CC/IC and CCF/ICF combinations
    for level1 in (results_dir).glob("*"):
        if not level1.is_dir():
            continue
        for level2 in level1.glob("*"):  # CC, IC
            if not level2.is_dir():
                continue
            for level3 in level2.glob("*"):  # CCF, ICF
                if not level3.is_dir():
                    continue
                cf_dir = level3 / "CF"
                if cf_dir.is_dir():
                    cf_dirs.append(cf_dir)
    pngs: List[Path] = []
    for d in cf_dirs:
        pngs.extend(sorted(d.glob("*.png")))
    return sorted(pngs)


def resolve_original(original_dir: Path, basename: str) -> Path:
    """
    Originals are under:
      Original/Correct/<basename> or Original/Incorrect/<basename>
    """
    p = original_dir / "Correct" / basename
    if p.is_file():
        return p
    p = original_dir / "Incorrect" / basename
    if p.is_file():
        return p
    raise FileNotFoundError(f"Original image not found for {basename}")


def main():
    parser = argparse.ArgumentParser(description="Flatten TIME results into paired images_/inpaint_ files.")
    parser.add_argument("--source", required=True, help="Path to the base results directory passed as --output_path in generate-ce.py")
    parser.add_argument("--exp_name", required=True, help="Experiment name used with generate-ce.py (--exp_name)")
    parser.add_argument("--output_path", required=True, help="Destination directory for flattened outputs")
    parser.add_argument("--limit", type=int, default=-1, help="Optional: limit number of pairs to export")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output_path).resolve()
    results_root = source / "Results" / args.exp_name
    original_root = source / "Original"

    if not results_root.is_dir():
        raise FileNotFoundError(f"Results folder not found: {results_root}")
    if not original_root.is_dir():
        raise FileNotFoundError(f"Original folder not found: {original_root}")

    output.mkdir(parents=True, exist_ok=True)

    # Gather CF images and sort by filename (so indices increase)
    cf_pngs = find_cf_images(results_root)
    if args.limit > -1:
        cf_pngs = cf_pngs[:args.limit]

    count = 0
    for idx, cf_path in enumerate(cf_pngs, start=1):
        basename = cf_path.name  # e.g., 0001234.png
        try:
            orig_path = resolve_original(original_root, basename)
        except FileNotFoundError:
            # Skip if we can't find the matching original
            continue

        images_name = f"images_{idx:05d}.png"
        inpaint_name = f"inpaint_{idx:05d}.png"

        shutil.copyfile(orig_path, output / images_name)
        shutil.copyfile(cf_path, output / inpaint_name)
        count += 1

    print(f"Exported {count} pairs to {output}")


if __name__ == "__main__":
    main()


