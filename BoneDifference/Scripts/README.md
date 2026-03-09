# BoneDifference Batch Processing Scripts

This folder contains helper scripts for running the **BoneDifference**
Slicer module in **headless / batch mode**.\
The scripts are designed for reproducible large‑scale processing of
datasets where each case is stored in a separate folder.

Typical dataset layout:

    DATASET_PATH/
      case_001/
        preop.nii.gz
        postop.nii.gz
      case_002/
        preop.nii.gz
        postop.nii.gz
      ...

The script will iterate over all `{CASE_ID}` directories inside
`DATASET_PATH`, run the BoneDifference pipeline, and save outputs into
the same case directory.

------------------------------------------------------------------------

# Usage

The script must be run through **3D Slicer** in headless mode.

General pattern:

    /path/to/Slicer --no-main-window \
      --python-script /path/to/repo/BoneDifference/scripts/process_files.py -- \
      [parameters]

The `--` separator is important because it prevents Slicer from
consuming the script arguments.

------------------------------------------------------------------------

# Example 1 --- Full pipeline with registration (recommended)

This example performs:

1.  BRAINSFit registration (postop → preop)
2.  subtraction (diff)
3.  binarization

and saves:

-   `registered.nii.gz` -- transformed postoperative scan
-   `diff.nii.gz` -- subtraction result
-   `mask.nii.gz` -- final binarized defect mask

```{=html}
<!-- -->
```
    /path/to/Slicer --no-main-window \
      --python-script /path/to/repo/BoneDifference/scripts/process_files.py -- \
      --dataset-path /path/to/dataset \
      --preop preop.nii.gz \
      --postop postop.nii.gz \
      --registered registered.nii.gz \
      --diff diff.nii.gz \
      --output mask.nii.gz \
      --registration-transform-type rigid_affine

### Important

This configuration **with registration enabled and default parameters**
is the **exact configuration used in our article** and is therefore
**recommended for reproducibility**.

------------------------------------------------------------------------

# Example 2 --- Already coregistered data (no registration)

If the postoperative scan is already registered to the preoperative
scan, registration can be skipped to save time significantly.

In this case the script only performs:

1.  subtraction
2.  binarization

and saves only the final mask. Saving of the subtraction result can also be enabled by providing the `--diff` param.

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

------------------------------------------------------------------------

# Output files

Depending on parameters, the following files may be generated inside
each case directory:

  -------------------------------------------------------------------------
  File                  Description
  --------------------- ---------------------------------------------------
  `registered.nii.gz`   Postoperative CT registered to the preoperative
                        scan

  `diff.nii.gz`         Subtraction result (`preop - registered`)

  `mask.nii.gz`         Final binarized skull defect mask
  
  -------------------------------------------------------------------------

If `mask.nii.gz` already exists, the case is **automatically skipped**.

------------------------------------------------------------------------

# Notes

-   Progress is displayed using **tqdm**.
-   The progress bar counts **only cases that require processing**, not
    skipped cases.
-   Errors can optionally be ignored using `--continue-on-error`.

------------------------------------------------------------------------

# Reproducibility

For reproducible results consistent with the publication:

-   use **Example 1**
-   keep **all default parameters**
-   enable **registration (`rigid_affine`)**

This ensures identical preprocessing, registration, and binarization
behavior as described in the article.
