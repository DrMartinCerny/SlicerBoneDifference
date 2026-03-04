import logging
import slicer


def register_postop_to_preop_affine(
    *,
    fixed_preop_node,
    moving_postop_node,
    output_transformed_postop_node,
    sampling_percentage: float = 0.2,
    initialize_transform_mode: str = "useGeometryAlign",
    transform_type: str = "Rigid+Affine",   # "Rigid" or "Rigid+Affine"
    number_of_iterations: int = 1500,
):
    """
    Run BRAINSFit registration with:
      fixed  = preop
      moving = postop
    Produces output volume in fixed (preop) space.

    Parameters
    ----------
    transform_type:
        "Rigid" or "Rigid+Affine" (default).
    initialize_transform_mode:
        BRAINSFit initializeTransformMode string (e.g., useGeometryAlign, useCenterOfHeadAlign, ...).
    number_of_iterations:
        BRAINSFit numberOfIterations (default 1500).
    """
    if fixed_preop_node is None or moving_postop_node is None:
        raise ValueError("fixed_preop_node and moving_postop_node must be set")
    if output_transformed_postop_node is None:
        raise ValueError("output_transformed_postop_node must be set")

    if not hasattr(slicer.modules, "brainsfit"):
        raise RuntimeError(
            "BRAINSFit module not found in this Slicer installation (slicer.modules.brainsfit missing)."
        )

    tt = (transform_type or "").strip()
    if tt not in ("Rigid", "Rigid+Affine"):
        raise ValueError(f"transform_type must be 'Rigid' or 'Rigid+Affine', got: {transform_type!r}")

    useRigid = True
    useAffine = (tt == "Rigid+Affine")

    params = {
        "fixedVolume": fixed_preop_node.GetID(),
        "movingVolume": moving_postop_node.GetID(),
        "outputVolume": output_transformed_postop_node.GetID(),

        "useRigid": bool(useRigid),
        "useAffine": bool(useAffine),

        "initializeTransformMode": str(initialize_transform_mode),

        "samplingPercentage": float(sampling_percentage),
        "numberOfIterations": int(number_of_iterations),
    }

    logging.info(f"Running BRAINSFit with params: {params}")

    cliNode = slicer.cli.run(
        slicer.modules.brainsfit,
        None,
        params,
        wait_for_completion=True,
    )

    status = cliNode.GetStatusString() if cliNode else "Unknown"
    if cliNode and cliNode.GetStatus() != cliNode.Completed:
        err = cliNode.GetErrorText() if hasattr(cliNode, "GetErrorText") else ""
        raise RuntimeError(f"BRAINSFit failed (status={status}). {err}")

    logging.info(f"BRAINSFit completed (status={status}).")
    return cliNode