import numpy as np
from scipy import ndimage as ndi


def _keep_largest_component(mask_zyx: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Keep only largest connected component in a boolean 3D mask."""
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    lab, n = ndi.label(mask_zyx, structure=structure)
    if n == 0:
        return mask_zyx
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return lab == largest


def _binary_closing(mask_zyx: np.ndarray, radius_xyz=(1, 1, 1)) -> np.ndarray:
    """Binary closing with a rectangular structuring element."""
    rx, ry, rz = map(int, radius_xyz)
    structure = np.ones((2 * rz + 1, 2 * ry + 1, 2 * rx + 1), dtype=bool)  # (z,y,x)
    return ndi.binary_closing(mask_zyx, structure=structure)


def _head_cap_mask(shape_zyx, spacing_xyz, direction_3x3, head_cap_depth_mm: float) -> np.ndarray:
    """
    Mask keeping only the top head_cap_depth_mm from vertex along superior-inferior axis.
    direction_3x3 is 3x3 with columns = i/j/k axis direction vectors (in any world frame),
    we use the Z component magnitude to decide SI axis and sign to decide which end is 'top'.
    """
    nz, ny, nx = shape_zyx
    size_ijk = (nx, ny, nz)
    sx, sy, sz = map(float, spacing_xyz)

    D = np.asarray(direction_3x3, dtype=np.float64)
    if D.shape != (3, 3):
        raise ValueError("direction_3x3 must be 3x3")

    zcomp = np.abs(D[2, :])
    si_axis = int(np.argmax(zcomp))     # 0=i,1=j,2=k
    si_dir = float(D[2, si_axis])       # sign

    spacings = (sx, sy, sz)
    n_slices = int(size_ijk[si_axis])
    depth_slices = int(head_cap_depth_mm / spacings[si_axis])
    depth_slices = max(1, min(depth_slices, n_slices))

    if si_dir > 0:
        start_slice = max(0, n_slices - depth_slices)
        end_slice = n_slices - 1
    else:
        start_slice = 0
        end_slice = min(depth_slices - 1, n_slices - 1)

    axis_map = {0: 2, 1: 1, 2: 0}  # i->x->2, j->y->1, k->z->0
    np_axis = axis_map[si_axis]

    cap = np.zeros(shape_zyx, dtype=bool)
    sl = [slice(None), slice(None), slice(None)]
    sl[np_axis] = slice(start_slice, end_slice + 1)
    cap[tuple(sl)] = True
    return cap


def binarize_diff(
    diff_zyx: np.ndarray,
    preop_zyx: np.ndarray,
    *,
    delta_hu_threshold: float,          # REQUIRED (GUI slider)
    bone_threshold_hu: float = 300.0,
    min_component_size: int = 50,
    head_cap_depth_mm: float = 100.0,
    keep_largest_bone_component: bool = True,
    closing_radius_xyz=(1, 1, 1),
    spacing_xyz=(1.0, 1.0, 1.0),
    direction_3x3=((1.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0)),
    connectivity: int = 1,
) -> np.ndarray:
    """
    Deterministic burr-hole candidate binarization.

    Inputs:
      - diff_zyx: float (z,y,x)  preop - postop_registered
      - preop_zyx: float (z,y,x) HU
      - delta_hu_threshold: float threshold applied to diff restricted to bone

    Output:
      - uint8 (z,y,x) mask 0/1
    """
    diff = np.asarray(diff_zyx, dtype=np.float32)
    preop = np.asarray(preop_zyx, dtype=np.float32)

    # 1) bone mask from preop
    bone = preop > float(bone_threshold_hu)
    if keep_largest_bone_component:
        bone = _keep_largest_component(bone, connectivity=connectivity)

    # 2) keep only positive differences (bone lost)
    diff_pos = np.maximum(diff, 0.0)

    # 3) restrict to bone region
    diff_bone = diff_pos * bone.astype(np.float32)

    # 4) hard threshold
    cand = diff_bone >= float(delta_hu_threshold)

    # 5a) closing
    if closing_radius_xyz is not None:
        cand = _binary_closing(cand, radius_xyz=closing_radius_xyz)

    # 5b) remove tiny CCs
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    lab, n = ndi.label(cand, structure=structure)
    if n > 0:
        counts = np.bincount(lab.ravel())
        keep = counts >= int(min_component_size)
        keep[0] = False
        cleaned = keep[lab]
    else:
        cleaned = cand

    # 6) head cap restriction
    cap = _head_cap_mask(cleaned.shape, spacing_xyz, direction_3x3, float(head_cap_depth_mm))
    final = cleaned & cap

    return final.astype(np.uint8)