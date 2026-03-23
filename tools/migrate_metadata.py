#!/usr/bin/env python3
"""Patch existing model metadata.json files for v0.3.8 compatibility.

Adds two fields that older metadata files may lack:

1. architecture.effective_input_channels -- the actual model input size
   (doubled when context_scale > 1)

2. input_config.normalization.channel_stats + precomputed -- embeds the
   training dataset normalization stats into input_config so the metadata
   is self-contained for normalization

Usage:
    python migrate_metadata.py                     # scan default model dir
    python migrate_metadata.py /path/to/models/    # scan specific directory
    python migrate_metadata.py --dry-run            # preview without writing
"""

import argparse
import json
import sys
from pathlib import Path


def patch_metadata(metadata_path, dry_run=False):
    """Patch a metadata.json to add missing v0.3.8 fields.

    Returns True if the file was (or would be) modified.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    changed = False
    model_name = metadata_path.parent.name
    label = "[DRY RUN] " if dry_run else ""

    # --- 1. Add effective_input_channels ---
    arch = metadata.get("architecture", {})
    if "effective_input_channels" not in arch:
        base_ch = int(arch.get("input_channels",
                               metadata.get("input_config", {}).get(
                                   "num_channels", 3)))
        ctx_scale = int(arch.get("context_scale", 1))
        effective = base_ch * 2 if ctx_scale > 1 else base_ch
        arch["effective_input_channels"] = effective
        metadata["architecture"] = arch
        changed = True
        print(f"  {label}{model_name}: effective_input_channels={effective} "
              f"(base={base_ch}, ctx={ctx_scale})")

    # --- 2. Embed normalization_stats into input_config ---
    norm_stats = metadata.get("normalization_stats")
    input_config = metadata.get("input_config", {})
    norm_config = input_config.get("normalization", {})
    if norm_stats and "channel_stats" not in norm_config:
        norm_config["precomputed"] = True
        norm_config["channel_stats"] = norm_stats
        input_config["normalization"] = norm_config
        metadata["input_config"] = input_config
        changed = True
        print(f"  {label}{model_name}: embedded {len(norm_stats)} channel "
              f"stats into input_config.normalization")

    if changed and not dry_run:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Patch model metadata for v0.3.8 compatibility")
    parser.add_argument("model_dir", nargs="?",
                        default=Path.home() / ".dlclassifier" / "models",
                        help="Root directory containing model subdirectories "
                        "(default: ~/.dlclassifier/models/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without writing")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Directory not found: {model_dir}")
        sys.exit(1)

    metadata_files = sorted(model_dir.glob("*/metadata.json"))
    if not metadata_files:
        print(f"No metadata.json files found in {model_dir}")
        sys.exit(0)

    print(f"Scanning {len(metadata_files)} models in {model_dir}")
    patched = 0
    for mf in metadata_files:
        try:
            if patch_metadata(mf, dry_run=args.dry_run):
                patched += 1
        except Exception as e:
            print(f"  ERROR {mf.parent.name}: {e}", file=sys.stderr)

    action = "would patch" if args.dry_run else "patched"
    print(f"\nDone: {action} {patched}/{len(metadata_files)} models")


if __name__ == "__main__":
    main()
