import logging
import slicer


def register_postop_to_preop_affine(
    *,
    fixed_preop_node,
    moving_postop_node,
    output_transformed_postop_node,
    sampling_percentage: float = 0.2,
    initialize_transform_mode: str = "useGeometryAlign",
):
    """
    Run BRAINSFit affine registration with:
      fixed  = preop
      moving = postop
    Produces output volume in fixed (preop) space.

    Notes:
    - This returns the CLI node (can be used to inspect status/params).
    - output_transformed_postop_node will be filled by BRAINSFit.
    """
    if fixed_preop_node is None or moving_postop_node is None:
        raise ValueError("fixed_preop_node and moving_postop_node must be set")
    if output_transformed_postop_node is None:
        raise ValueError("output_transformed_postop_node must be set")

    # Ensure BRAINSFit is available
    if not hasattr(slicer.modules, "brainsfit"):
        raise RuntimeError("BRAINSFit module not found in this Slicer installation (slicer.modules.brainsfit missing).")

    # BRAINSFit parameters (CLI)
    params = {
        "fixedVolume": fixed_preop_node.GetID(),
        "movingVolume": moving_postop_node.GetID(),
        "outputVolume": output_transformed_postop_node.GetID(),

        # Rigid + Affine (affine includes rigid stage typically; we enable both explicitly)
        "useRigid": True,
        "useAffine": True,

        # Initialization (important when headers are unreliable)
        "initializeTransformMode": initialize_transform_mode,

        # Speed/robustness knobs
        "samplingPercentage": float(sampling_percentage),
    }

    logging.info(f"Running BRAINSFit with params: {params}")

    # Run synchronously (wait_for_completion=True)
    cliNode = slicer.cli.run(
        slicer.modules.brainsfit,
        None,
        params,
        wait_for_completion=True,
    )

    status = cliNode.GetStatusString() if cliNode else "Unknown"
    if cliNode and cliNode.GetStatus() != cliNode.Completed:
        # Try to surface useful error text
        err = cliNode.GetErrorText() if hasattr(cliNode, "GetErrorText") else ""
        raise RuntimeError(f"BRAINSFit failed (status={status}). {err}")

    logging.info(f"BRAINSFit completed (status={status}).")
    return cliNode