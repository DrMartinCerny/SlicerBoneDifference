#!/usr/bin/env python3
"""
Headless batch runner for the BoneDifference Slicer scripted module.

Dataset layout:
  DATASET_PATH/
    CASE_ID_1/
      preop.nii.gz
      postop.nii.gz
      ...
    CASE_ID_2/
      ...

The script crawls DATASET_PATH/{CASE_ID}/, loads preop+postop, runs BoneDifference,
optionally saves transformed postop and/or diff (if filenames provided), and always
saves the binarized mask to --output.

Key behavior:
- Skips a case if {CASE_ID}/{output} already exists.
- Registration control is via one flag:
    --registration-transform-type {none,rigid,rigid_affine}
  If 'none', NO registration is performed and the script always uses --postop as input
  (it does NOT load --registered as input).

Examples:

1) Run registration and save transformed postop + diff + mask:
  /path/to/Slicer --no-main-window \
    --python-script /path/to/repo/BoneDifference/scripts/process_files.py -- \
    --dataset-path /path/to/dataset \
    --preop preop.nii.gz \
    --postop postop.nii.gz \
    --registered registered.nii.gz \
    --diff diff.nii.gz \
    --output mask.nii.gz \
    --registration-transform-type rigid_affine

2) Already coregistered (no registration), save only mask:
  /path/to/Slicer --no-main-window \
    --python-script /path/to/repo/BoneDifference/scripts/process_files.py -- \
    --dataset-path /path/to/dataset \
    --preop preop.nii.gz \
    --postop postop.nii.gz \
    --output mask.nii.gz \
    --registration-transform-type none \
    --registration-number-of-iterations 1500 \
    --registration-initialize-transform-mode useGeometryAlign \
    --delta-hu-threshold 700 \
    --bone-threshold-hu 300 \
    --min-component-size 50 \
    --head-cap-depth-mm 100 \
    --keep-largest-bone-component
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# -------------------------------------------------------------------------
# Ensure we can import BoneDifference.py when running from scripts/
# -------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPTS_DIR.parent  # .../BoneDifference
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import slicer  # noqa: E402
import BoneDifference  # noqa: E402


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-process a dataset of CT cases with BoneDifference in headless Slicer."
    )

    # Required structure args
    p.add_argument("--dataset-path", required=True, help="Path to dataset root containing case folders.")
    p.add_argument("--preop", required=True, help="Preop filename inside each case folder (e.g., preop.nii.gz).")
    p.add_argument("--postop", required=True, help="Postop filename inside each case folder (e.g., postop.nii.gz).")
    p.add_argument("--output", required=True, help="Output mask filename inside each case folder (e.g., mask.nii.gz).")

    # Optional filenames to SAVE (never used as input when reg type == none)
    p.add_argument(
        "--registered",
        default=None,
        help=(
            "Optional filename inside each case folder to save transformed postop (only if registration is enabled). "
            "Ignored when --registration-transform-type none. If omitted, transformed postop is not saved."
        ),
    )
    p.add_argument(
        "--diff",
        default=None,
        help="Optional diff filename to save inside each case folder (e.g., diff.nii.gz). If omitted, diff is not saved.",
    )

    # Registration control
    p.add_argument(
        "--registration-transform-type",
        default="rigid_affine",
        choices=["none", "rigid", "rigid_affine"],
        help=(
            "Registration mode: "
            "'none' = assume already coregistered (but still loads --postop; strict geometry check enforces voxel match). "
            "'rigid' = BRAINSFit rigid. "
            "'rigid_affine' = BRAINSFit rigid+affine."
        ),
    )
    p.add_argument(
        "--registration-initialize-transform-mode",
        default="useGeometryAlign",
        choices=[
            "useGeometryAlign",
            "useCenterOfHeadAlign",
            "useCenterOfROIAlign",
            "useMomentsAlign",
            "useNone",
        ],
        help="BRAINSFit initialization strategy.",
    )
    p.add_argument(
        "--registration-number-of-iterations",
        type=int,
        default=1500,
        help="BRAINSFit number of iterations (used when registration is enabled).",
    )

    # Binarization params (match GUI defaults)
    p.add_argument("--delta-hu-threshold", type=float, default=700.0, help="Delta HU threshold.")
    p.add_argument("--bone-threshold-hu", type=float, default=300.0, help="Bone HU threshold for skull mask.")
    p.add_argument("--min-component-size", type=int, default=50, help="Minimum component size in voxels.")
    p.add_argument("--head-cap-depth-mm", type=float, default=100.0, help="Head cap depth in mm.")
    p.add_argument(
        "--keep-largest-bone-component",
        action="store_true",
        default=True,
        help="Keep only largest bone component (skull). Default: True.",
    )
    p.add_argument(
        "--no-keep-largest-bone-component",
        dest="keep_largest_bone_component",
        action="store_false",
        help="Disable keeping only largest bone component.",
    )

    # Logging / control
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, errors in one case won't stop the whole run (default: stop on first error).",
    )

    return p


# -------------------------------------------------------------------------
# Dataset discovery
# -------------------------------------------------------------------------
def _list_case_dirs(dataset_path: Path) -> List[Path]:
    """Return direct subdirectories of dataset_path as case folders."""
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found or not a directory: {dataset_path}")
    return sorted([p for p in dataset_path.iterdir() if p.is_dir()])


def _plan_work(
    case_dirs: List[Path],
    *,
    preop_name: str,
    postop_name: str,
    output_name: str,
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Build:
      - to_process: cases where output does NOT exist AND inputs exist
      - skipped: list of (case_dir, reason)
    """
    to_process: List[Path] = []
    skipped: List[Tuple[Path, str]] = []

    for case_dir in case_dirs:
        out_path = case_dir / output_name
        if out_path.exists():
            skipped.append((case_dir, f"output exists: {out_path.name}"))
            continue

        preop_path = case_dir / preop_name
        postop_path = case_dir / postop_name

        if not preop_path.exists():
            skipped.append((case_dir, f"missing preop: {preop_path.name}"))
            continue
        if not postop_path.exists():
            skipped.append((case_dir, f"missing postop: {postop_path.name}"))
            continue

        to_process.append(case_dir)

    return to_process, skipped


# -------------------------------------------------------------------------
# Per-case processing
# -------------------------------------------------------------------------
def _remove_nodes(nodes):
    """Remove MRML nodes to keep memory stable across many cases."""
    for n in nodes:
        try:
            if n is not None:
                slicer.mrmlScene.RemoveNode(n)
        except Exception:
            pass


def _process_one_case(case_dir: Path, args: argparse.Namespace):
    """
    Process one case folder.

    - Always loads preop from {case}/{--preop}
    - Always loads postop from {case}/{--postop}
    - If registration-transform-type != none, performs BRAINSFit postop->preop and uses transformed for diff.
    - Saves:
        - mask always -> {case}/{--output}
        - diff if --diff provided
        - transformed postop if --registered provided AND registration is enabled
    """
    preop_path = case_dir / args.preop
    postop_path = case_dir / args.postop
    out_mask_path = case_dir / args.output
    out_diff_path = (case_dir / args.diff) if args.diff else None
    out_registered_path = (case_dir / args.registered) if args.registered else None

    # Load inputs
    preopNode = slicer.util.loadVolume(str(preop_path))
    if preopNode is None:
        raise RuntimeError(f"Failed to load preop: {preop_path}")

    postopNode = slicer.util.loadVolume(str(postop_path))
    if postopNode is None:
        raise RuntimeError(f"Failed to load postop: {postop_path}")

    # Output nodes (created in-scene)
    diffNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Diff")
    maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Mask")

    # Registration control
    reg_mode = args.registration_transform_type
    do_registration = reg_mode in ("rigid", "rigid_affine")

    transformedNode = None
    transform_type_str = None

    if do_registration:
        transformedNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "postop_transformed_to_preop"
        )
        transform_type_str = "Rigid" if reg_mode == "rigid" else "Rigid+Affine"

    # Run module logic
    logic = BoneDifference.BoneDifferenceLogic()
    logic.run(
        preopNode=preopNode,
        postopNode=postopNode,
        diffNode=diffNode,
        maskNode=maskNode,

        do_registration=bool(do_registration),
        transformed_postop_node=transformedNode,

        # registration params (used only if do_registration=True)
        registration_transform_type=str(transform_type_str) if transform_type_str else "Rigid+Affine",
        registration_initialize_transform_mode=str(args.registration_initialize_transform_mode),
        registration_number_of_iterations=int(args.registration_number_of_iterations),

        # binarization params
        delta_hu_threshold=float(args.delta_hu_threshold),
        bone_threshold_hu=float(args.bone_threshold_hu),
        min_component_size=int(args.min_component_size),
        head_cap_depth_mm=float(args.head_cap_depth_mm),
        keep_largest_bone_component=bool(args.keep_largest_bone_component)
    )

    # Save transformed postop if requested AND registration was performed
    if do_registration and out_registered_path is not None:
        if not slicer.util.saveNode(transformedNode, str(out_registered_path)):
            raise RuntimeError(f"Failed to save transformed postop: {out_registered_path}")

    # Save diff if requested
    if out_diff_path is not None:
        if not slicer.util.saveNode(diffNode, str(out_diff_path)):
            raise RuntimeError(f"Failed to save diff: {out_diff_path}")

    # Save mask (required)
    if not slicer.util.saveNode(maskNode, str(out_mask_path)):
        raise RuntimeError(f"Failed to save mask: {out_mask_path}")

    # Cleanup
    _remove_nodes([preopNode, postopNode, transformedNode, diffNode, maskNode])


# -------------------------------------------------------------------------
def main():
    parser = _build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    dataset_path = Path(args.dataset_path).expanduser().resolve()

    # Discover cases
    case_dirs = _list_case_dirs(dataset_path)

    # Plan work so tqdm total excludes skipped cases
    to_process, skipped = _plan_work(
        case_dirs,
        preop_name=args.preop,
        postop_name=args.postop,
        output_name=args.output,
    )

    logging.info(f"Found {len(case_dirs)} case folders.")
    logging.info(f"Will process {len(to_process)} cases (skipping {len(skipped)}).")

    if skipped and args.verbose:
        for cd, reason in skipped[:50]:
            logging.debug(f"Skip {cd.name}: {reason}")
        if len(skipped) > 50:
            logging.debug(f"... and {len(skipped) - 50} more skipped cases")

    errors: List[Tuple[str, str]] = []

    iterator = to_process
    if tqdm is not None:
        iterator = tqdm(to_process, desc="BoneDifference", unit="case")

    for case_dir in iterator:
        try:
            _process_one_case(case_dir, args)
        except Exception as e:
            msg = f"{case_dir.name}: {e}"
            logging.exception(msg)
            errors.append((case_dir.name, str(e)))

            if not args.continue_on_error:
                break

    if errors:
        logging.error(f"Finished with {len(errors)} error(s). Showing first 20:")
        for cid, emsg in errors[:20]:
            logging.error(f"  - {cid}: {emsg}")
    else:
        logging.info("Finished successfully with no errors.")

    slicer.util.exit()


if __name__ == "__main__":
    main()