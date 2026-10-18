"""
registration_v2 -- postop -> preop registration for burr-hole bone-defect detection.

What is different from v1 (single BRAINSFit call, geometry-align init, no masks):

 1. Initialisation does NOT use the image geometry. The head is located in both scans from the
    axial head-area profile along the superior-inferior axis (robust to a long neck and to
    truncated field of views) and, when the vertex is visible in both scans, from the skull vertex.
    In-plane position comes from the centroid of the cranial slab only (never the neck).
 2. Two rigid BRAINSFit stages, both with ROI masks, so couch / headrest / neck / soft-tissue
    changes cannot pull the solution:
        stage 1: cranial head mask (top CRANIAL_DEPTH_MM of the head)
        stage 2: dilated skull-bone mask, finer step lengths, started from stage 1
    (transform_type "Rigid+Affine" adds a third affine stage on the skull ROI.)
 3. Voxels of the registered postop that lie outside the postop field of view are filled with
    SENTINEL (-2000 HU) instead of 0 HU, so BoneDifference can exclude them from the difference.
 4. Three candidate initialisations (head-based +t, head-based -t, BRAINSFit geometry align) are probed with a
    1-iteration BRAINSFit run each and the one that aligns the skull best is used. This protects against
    a wrong sign convention or a failed head localisation.
 5. Before the final stage the postop is low-pass filtered to the (coarser) preop resolution, mainly along z
    (e.g. preop 4 mm slices vs postop 0.5 mm), so partial-volume edges do not show up as "removed bone".

The numerical helpers are plain numpy/scipy; only register_postop_to_preop_affine() touches Slicer.
"""
import logging

import numpy as np
import slicer
import vtk
from scipy import ndimage as ndi

SENTINEL = -2000.0        # fill value for "outside the postop field of view"
HEAD_HU = -300.0          # head / air threshold
BONE_HU = 300.0           # bone threshold (same as binarize default)
CRANIAL_DEPTH_MM = 130.0  # ROI covers this much of the head below the vertex (skull base is ~100-110 mm below it)
SKULL_DILATE_MM = 6.0     # stage-2 ROI: bone dilated by this much
PROFILE_SEARCH_MM = 250   # +-search range of the z-shift estimate
VERTEX_TRUST_MM = 25.0    # vertex-based z-shift is used only if it agrees with the profile estimate within this
ANTIALIAS = True          # blur the postop to the preop resolution where the postop is finer (see docstring, point 5)

_AX = {0: 2, 1: 1, 2: 0}  # ijk axis -> numpy axis of an array in (k, j, i) = (z, y, x) order


def _log(msg):
    print(msg, flush=True)  # Slicer's python logging is not shown in headless runs


# ------------------------------------------------------------------------------------------------
# geometry helpers (pure numpy)
# ------------------------------------------------------------------------------------------------
def ijk_to_ras_matrix(node) -> np.ndarray:
    m = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(m)
    return np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=np.float64)


def spacing_zyx_from_matrix(M: np.ndarray):
    n = np.linalg.norm(M[:3, :3], axis=0)  # (i, j, k)
    return np.array([n[2], n[1], n[0]], dtype=np.float64)


def si_axis_info(M: np.ndarray):
    """(ijk axis most aligned with superior, +1/-1 = index increases toward superior, signed S-step per index)"""
    D = M[:3, :3]
    a = int(np.argmax(np.abs(D[2, :])))
    return a, (1.0 if D[2, a] >= 0 else -1.0), float(D[2, a])


def slice_s_coords(M: np.ndarray, shape_zyx):
    """S (superior) coordinate in mm of every slice along the SI array axis (axial acquisitions)."""
    a, _, step = si_axis_info(M)
    dims_ijk = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
    s0 = M[2, 3] + sum(M[2, c] * (dims_ijk[c] - 1) / 2.0 for c in range(3) if c != a)
    return s0 + step * np.arange(dims_ijk[a])


def _largest_cc(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    return lab == cnt.argmax()


def head_mask(arr: np.ndarray, spacing_zyx, thr: float = HEAD_HU, open_mm: float = 8.0) -> np.ndarray:
    """Head without couch / headrest: HU > thr, opening (cuts thin supports), largest component, filled.
    Computed on a 2x downsampled grid, returned on the full grid."""
    ds = arr[::2, ::2, ::2] > thr
    sp2 = np.asarray(spacing_zyx, dtype=np.float64) * 2.0
    r = np.maximum(1, np.round(open_mm / sp2 / 2.0)).astype(int)
    m = ndi.binary_opening(ds, np.ones(tuple(2 * r + 1), bool))
    if not m.any():
        m = ds
    m = ndi.binary_fill_holes(_largest_cc(m))
    full = np.repeat(np.repeat(np.repeat(m, 2, 0), 2, 1), 2, 2)[: arr.shape[0], : arr.shape[1], : arr.shape[2]]
    if full.shape != arr.shape:
        full = np.pad(full, [(0, a - b) for a, b in zip(arr.shape, full.shape)])
    return full


def _other_axes(ax):
    return tuple(i for i in range(3) if i != ax)


def head_profile(head: np.ndarray, M: np.ndarray, spacing_zyx):
    """(S coordinates ascending, equivalent head diameter per axial slice)"""
    a, _, _ = si_axis_info(M)
    ax = _AX[a]
    other = _other_axes(ax)
    area = head.sum(axis=other).astype(np.float64) * float(np.prod([spacing_zyx[i] for i in other]))
    s = slice_s_coords(M, head.shape)
    order = np.argsort(s)
    return s[order], np.sqrt(area[order])


def vertex_info(arr: np.ndarray, head: np.ndarray, M: np.ndarray, min_vox: int = 100):
    """(S of the skull vertex, vertex_is_cut_by_the_field_of_view) or None if no bone."""
    a, sgn, _ = si_axis_info(M)
    ax = _AX[a]
    other = _other_axes(ax)
    n = arr.shape[ax]
    bone_counts = ((arr > BONE_HU) & head).sum(axis=other)
    idx = np.where(bone_counts >= min_vox)[0]
    if len(idx) == 0:
        return None
    top = int(idx.max() if sgn > 0 else idx.min())
    end = n - 1 if sgn > 0 else 0
    head_counts = head.sum(axis=other)
    truncated = abs(top - end) <= 1 or head_counts[end] >= min_vox
    return float(slice_s_coords(M, arr.shape)[top]), bool(truncated)


def _profile_shift(sf, df, sm, dm, search=PROFILE_SEARCH_MM):
    """Integer-mm shift t (moving + t -> fixed) minimising the normalised L1 distance of the two
    diameter profiles over their overlap."""
    gf = np.arange(np.ceil(sf.min()), np.floor(sf.max()) + 1)
    gm = np.arange(np.ceil(sm.min()), np.floor(sm.max()) + 1)
    f = np.interp(gf, sf, df)
    m = np.interp(gm, sm, dm)
    min_ov = min(60.0, 0.5 * min(len(f), len(m)))
    best, best_t = np.inf, 0
    for t in range(-search, search + 1):
        lo = max(gf[0], gm[0] + t)
        hi = min(gf[-1], gm[-1] + t)
        if hi - lo + 1 < min_ov:
            continue
        a = f[int(lo - gf[0]): int(hi - gf[0]) + 1]
        b = m[int(lo - t - gm[0]): int(hi - t - gm[0]) + 1]
        c = np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-9)
        if c < best:
            best, best_t = c, t
    return float(best_t), float(best)


def slab_slices(shape_zyx, M, s_lo, s_hi):
    """numpy index tuple selecting the slices whose S lies in [s_lo, s_hi] (None if empty)."""
    a, _, _ = si_axis_info(M)
    ax = _AX[a]
    s = slice_s_coords(M, shape_zyx)
    idx = np.where((s >= s_lo) & (s <= s_hi))[0]
    if len(idx) == 0:
        return None
    sl = [slice(None)] * 3
    sl[ax] = slice(int(idx.min()), int(idx.max()) + 1)
    return tuple(sl)


def _centroid_ras(mask_sub: np.ndarray, offset_zyx, M: np.ndarray, step: int = 4):
    pts = np.argwhere(mask_sub[::step, ::step, ::step]) * step + np.asarray(offset_zyx)
    if len(pts) == 0:
        return None
    c = pts.mean(axis=0)  # (k, j, i)
    return (M @ np.array([c[2], c[1], c[0], 1.0]))[:3]


def cranial_s_hi(vertex_f, sf, sm, t_s):
    """Top (S, fixed space) of the cranial slab: the preop vertex if it is visible, else the top of the common range."""
    top_common = min(sf.max(), sm.max() + t_s)
    if vertex_f is not None and not vertex_f[1] and vertex_f[0] <= top_common + 1:
        return float(vertex_f[0])
    return float(top_common)


def estimate_initial_translation(pre, M_f, sp_f, head_f, post, M_m, sp_m, head_m):
    """Translation t (RAS, mm) such that  moving + t  lies on the fixed head. Returns (t, info)."""
    sf, df = head_profile(head_f, M_f, sp_f)
    sm, dm = head_profile(head_m, M_m, sp_m)
    t_prof, cost = _profile_shift(sf, df, sm, dm)
    vf, vm = vertex_info(pre, head_f, M_f), vertex_info(post, head_m, M_m)
    t_s, how = t_prof, "profile"
    if vf and vm and not vf[1] and not vm[1]:
        t_vert = vf[0] - vm[0]
        if abs(t_vert - t_prof) <= VERTEX_TRUST_MM:
            t_s, how = t_vert, "vertex"
    # in-plane: centroid of the cranial slab common to both scans (the neck is excluded by construction)
    s_hi = cranial_s_hi(vf, sf, sm, t_s)
    lo_common = max(sf.min(), sm.min() + t_s)
    s_lo = max(s_hi - CRANIAL_DEPTH_MM, lo_common)
    sl_f = slab_slices(pre.shape, M_f, s_lo, s_hi)
    sl_m = slab_slices(post.shape, M_m, s_lo - t_s, s_hi - t_s)
    c_f = c_m = None
    if sl_f is not None and sl_m is not None:
        off_f = [s.start or 0 for s in sl_f]
        off_m = [s.start or 0 for s in sl_m]
        c_f = _centroid_ras(head_f[sl_f], off_f, M_f)
        c_m = _centroid_ras(head_m[sl_m], off_m, M_m)
    if c_f is None or c_m is None:  # no usable overlap: fall back to whole-head centroids
        c_f = _centroid_ras(head_f, [0, 0, 0], M_f)
        c_m = _centroid_ras(head_m, [0, 0, 0], M_m)
    t = np.array([c_f[0] - c_m[0], c_f[1] - c_m[1], t_s], dtype=np.float64)
    info = dict(z_method=how, z_profile=t_prof, z_vertex=(None if not (vf and vm) else vf[0] - vm[0]),
                profile_cost=cost, vertex_pre=vf, vertex_post=vm, s_hi=float(s_hi), sf_max=float(sf.max()), sm_max=float(sm.max()))
    return t, info


def cranial_rois(pre, M_f, sp_f, head_f, post, M_m, sp_m, head_m, t, s_hi):
    """(stage-1 ROI fixed, stage-1 ROI moving, stage-2 ROI fixed, stage-2 ROI moving) as bool arrays."""
    out = []
    for arr, M, sp, head, shift in ((pre, M_f, sp_f, head_f, 0.0), (post, M_m, sp_m, head_m, t[2])):
        slab = np.zeros(arr.shape, bool)
        sl = slab_slices(arr.shape, M, s_hi - CRANIAL_DEPTH_MM - shift, s_hi - shift)
        if sl is not None:
            slab[sl] = True
        roi1 = head & slab
        bone = (arr > BONE_HU) & roi1
        roi2 = np.zeros(arr.shape, bool)
        if bone.any():
            zz, yy, xx = np.where(bone.any(axis=(1, 2)))[0], np.where(bone.any(axis=(0, 2)))[0], np.where(bone.any(axis=(0, 1)))[0]
            mz = np.ceil(SKULL_DILATE_MM / sp / 1.0).astype(int) + 2
            lo = [max(0, int(v[0]) - mz[i]) for i, v in enumerate((zz, yy, xx))]
            hi = [min(arr.shape[i], int(v[-1]) + mz[i] + 1) for i, v in enumerate((zz, yy, xx))]
            box = tuple(slice(lo[i], hi[i]) for i in range(3))
            dist = ndi.distance_transform_edt(~bone[box], sampling=sp)
            roi2[box] = dist <= SKULL_DILATE_MM
            roi2 &= head
        out.append((roi1, roi2))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def head_top_cap_mask(preop_zyx, M, spacing_xyz, depth_mm, bone_hu=BONE_HU, keep_largest=True, min_vox=100):
    """Slab of depth_mm BELOW THE HIGHEST SKULL SLICE (v1 used the top end of the image, which is wrong
    whenever the scan extends above the vertex). Bool array, True inside the cap."""
    a, sgn, _ = si_axis_info(M)
    ax = _AX[a]
    n = preop_zyx.shape[ax]
    bone = preop_zyx > bone_hu
    if keep_largest:
        bone = _largest_cc(bone)
    counts = bone.sum(axis=_other_axes(ax))
    idx = np.where(counts >= min_vox)[0]
    if len(idx) == 0:
        idx = np.where(counts > 0)[0]
    cap = np.zeros(preop_zyx.shape, bool)
    if len(idx) == 0:
        cap[:] = True
        return cap
    depth = max(1, int(depth_mm / float(spacing_xyz[a])))
    sl = [slice(None)] * 3
    if sgn > 0:
        top = int(idx.max())
        sl[ax] = slice(max(0, top - depth + 1), top + 1)
    else:
        top = int(idx.min())
        sl[ax] = slice(top, min(n, top + depth))
    cap[tuple(sl)] = True
    return cap


def probe_translation(pre, M_f, head_f, post, M_m, t, max_pts=150000):
    """Score a candidate moving->fixed translation t (RAS mm) by how much of the preop skull lands on postop skull.
    Pure numpy. Returns (score, bone fraction, coverage)."""
    bone = _largest_cc(pre > BONE_HU)
    idx = np.argwhere(bone)
    if len(idx) > max_pts:
        idx = idx[np.random.default_rng(0).choice(len(idx), max_pts, replace=False)]
    ijk1 = np.c_[idx[:, 2], idx[:, 1], idx[:, 0], np.ones(len(idx))]       # x,y,z order
    ras = (M_f @ ijk1.T)
    ras[:3] -= np.asarray(t, float)[:, None]                                 # fixed -> moving
    ijk_m = (np.linalg.inv(M_m) @ ras)[:3]
    x, y, z = ijk_m
    sh = post.shape
    inside = (x >= 0) & (y >= 0) & (z >= 0) & (x <= sh[2] - 1) & (y <= sh[1] - 1) & (z <= sh[0] - 1)
    cov = float(inside.mean())
    if inside.sum() < 100:
        return 0.0, 0.0, cov
    vals = ndi.map_coordinates(post, [z[inside], y[inside], x[inside]], order=1, mode="nearest")
    frac = float((vals > BONE_HU).mean())
    return frac * np.sqrt(cov), frac, cov


def centre_ras(M, shape_zyx):
    return (M @ np.array([(shape_zyx[2] - 1) / 2.0, (shape_zyx[1] - 1) / 2.0, (shape_zyx[0] - 1) / 2.0, 1.0]))[:3]


def antialias_sigmas(M_f, M_m):
    """Gaussian sigma (voxels, numpy axis order z,y,x of the MOVING array) that brings the moving volume to the
    (coarser) resolution of the fixed volume. 0 where the moving volume is not finer."""
    Df = M_f[:3, :3] / np.linalg.norm(M_f[:3, :3], axis=0)
    Dm = M_m[:3, :3] / np.linalg.norm(M_m[:3, :3], axis=0)
    sp_f, sp_m = np.linalg.norm(M_f[:3, :3], axis=0), np.linalg.norm(M_m[:3, :3], axis=0)
    sig = np.zeros(3)
    for a in range(3):                                   # moving ijk axis a
        b = int(np.argmax(np.abs(Dm[:, a] @ Df)))         # fixed ijk axis it is parallel to
        if sp_f[b] > 1.2 * sp_m[a]:
            # slice profile ~ box of width sp_f: std = width / sqrt(12); subtract what the moving volume already has
            sig[_AX[a]] = (np.sqrt(sp_f[b] ** 2 - sp_m[a] ** 2) / np.sqrt(12.0)) / sp_m[a]
    return sig


def alignment_score(bone_f_s, out_s):
    """(score, skull Dice, coverage) of a probe output vs the preop skull. All arrays strided identically."""
    v = out_s > SENTINEL + 1.0
    inside = bone_f_s & v
    cov = float(inside.sum() / max(bone_f_s.sum(), 1))
    br = (out_s > BONE_HU) & v
    dice = float(2.0 * (inside & br).sum() / max(inside.sum() + br.sum(), 1))
    return dice * np.sqrt(cov), dice, cov


def quality_metrics(pre, reg, valid, spacing_zyx):
    """skull Dice / coverage / bone NCC of the final registration, for the log."""
    st = tuple(int(max(1, round(2.0 / s))) for s in spacing_zyx)
    sl = tuple(slice(None, None, k) for k in st)
    p, r, v = pre[sl], reg[sl], valid[sl]
    bone_p = _largest_cc(p > BONE_HU)
    score, dice, cov = alignment_score(bone_p, np.where(v, r, SENTINEL))
    head = (p > HEAD_HU) & v
    ncc = float("nan")
    if head.sum() > 100:
        a, b = np.clip(p[head], 0, 1500), np.clip(r[head], 0, 1500)
        if a.std() > 0 and b.std() > 0:
            ncc = float(np.corrcoef(a, b)[0, 1])
    return dict(skull_dice=dice, coverage=cov, bone_ncc=ncc, fov=float(v.mean()))


# ------------------------------------------------------------------------------------------------
# Slicer glue
# ------------------------------------------------------------------------------------------------
def _matrix_to_vtk(A: np.ndarray) -> vtk.vtkMatrix4x4:
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(A[r, c]))
    return m


def _transform_matrix(node) -> np.ndarray:
    m = vtk.vtkMatrix4x4()
    node.GetMatrixTransformToParent(m)
    return np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=np.float64)


def _add_label(arr_bool, M, name):
    return slicer.util.addVolumeFromArray(arr_bool.astype(np.uint8), ijkToRAS=M, name=name,
                                          nodeClassName="vtkMRMLLabelMapVolumeNode")


def _run_brainsfit(params, what):
    _log(f"[v2] BRAINSFit {what}")
    cli = slicer.cli.run(slicer.modules.brainsfit, None, params, wait_for_completion=True)
    if cli is None or cli.GetStatus() != cli.Completed:
        err = cli.GetErrorText() if (cli is not None and hasattr(cli, "GetErrorText")) else ""
        raise RuntimeError(f"BRAINSFit failed ({what}; status={cli.GetStatusString() if cli else 'None'}). {err}")
    return cli


def register_postop_to_preop_affine(
    *,
    fixed_preop_node,
    moving_postop_node,
    output_transformed_postop_node,
    sampling_percentage: float = 0.2,
    initialize_transform_mode: str = "useGeometryAlign",   # kept for CLI compatibility; ignored by v2
    transform_type: str = "Rigid+Affine",                  # "Rigid" or "Rigid+Affine"
    number_of_iterations: int = 1500,
):
    """Register moving = postop to fixed = preop. Output volume in preop space; voxels outside the
    postop field of view are SENTINEL."""
    if fixed_preop_node is None or moving_postop_node is None:
        raise ValueError("fixed_preop_node and moving_postop_node must be set")
    if output_transformed_postop_node is None:
        raise ValueError("output_transformed_postop_node must be set")
    if not hasattr(slicer.modules, "brainsfit"):
        raise RuntimeError("BRAINSFit module not found in this Slicer installation.")
    tt = (transform_type or "").strip()
    if tt not in ("Rigid", "Rigid+Affine"):
        raise ValueError(f"transform_type must be 'Rigid' or 'Rigid+Affine', got: {transform_type!r}")
    if initialize_transform_mode != "useGeometryAlign":
        _log(f"[v2] initialize_transform_mode={initialize_transform_mode!r} is ignored (head-based initialisation)")

    pre = slicer.util.arrayFromVolume(fixed_preop_node)
    post = slicer.util.arrayFromVolume(moving_postop_node)
    M_f, M_m = ijk_to_ras_matrix(fixed_preop_node), ijk_to_ras_matrix(moving_postop_node)
    sp_f, sp_m = spacing_zyx_from_matrix(M_f), spacing_zyx_from_matrix(M_m)

    head_f, head_m = head_mask(pre, sp_f), head_mask(post, sp_m)
    t_head, info = estimate_initial_translation(pre, M_f, sp_f, head_f, post, M_m, sp_m, head_m)
    t_geo = centre_ras(M_f, pre.shape) - centre_ras(M_m, post.shape)
    cands = {"head(+t)": t_head, "head(-t)": -t_head, "geometry": t_geo}
    scores = {k: probe_translation(pre, M_f, head_f, post, M_m, v) for k, v in cands.items()}
    best = max(scores, key=lambda k: scores[k][0])
    t = cands[best]
    _log("[v2] init candidates: " + " | ".join(f"{k}: score {v[0]:.3f} (bone {v[1]:.2f}, cov {v[2]:.2f}, t={np.round(cands[k], 1).tolist()})"
                                              for k, v in scores.items()) + f" -> {best}")
    _log(f"[v2] head estimate: {info}")
    if best != "head(+t)":
        _log(f"[v2] WARNING: initialisation '{best}' beat the head-based estimate (check this case)")
    if np.linalg.norm(t) > 200:
        _log(f"[v2] WARNING: very large initial translation {np.linalg.norm(t):.0f} mm")

    roi1_f, roi1_m, roi2_f, roi2_m = cranial_rois(pre, M_f, sp_f, head_f, post, M_m, sp_m, head_m, t, info["s_hi"])
    nodes = []
    try:
        def label_pair(rf, rm, tag):
            if rf.sum() < 20000 or rm.sum() < 20000:
                _log(f"[v2] WARNING: ROI '{tag}' too small ({int(rf.sum())}/{int(rm.sum())} vox); stage runs without mask")
                return None
            a, b = _add_label(rf, M_f, f"v2_{tag}_fixed"), _add_label(rm, M_m, f"v2_{tag}_moving")
            nodes.extend([a, b])
            return a, b

        masks1, masks2 = label_pair(roi1_f, roi1_m, "roi1"), label_pair(roi2_f, roi2_m, "roi2")
        del roi1_f, roi1_m, roi2_f, roi2_m, head_f, head_m

        moving_reg = moving_postop_node
        if ANTIALIAS:
            sig = antialias_sigmas(M_f, M_m)
            if sig.max() > 0:
                _log(f"[v2] low-pass filtering postop to preop resolution, sigma (voxels z,y,x) = {np.round(sig, 2).tolist()}")
                sm_arr = ndi.gaussian_filter(post.astype(np.float32), sig, mode="nearest")
                moving_reg = slicer.util.addVolumeFromArray(sm_arr, ijkToRAS=M_m, name="v2_postop_lp")
                nodes.append(moving_reg)
                del sm_arr

        tn = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "v2_init")
        nodes.append(tn)
        A = np.eye(4)
        A[:3, 3] = t
        tn.SetMatrixTransformToParent(_matrix_to_vtk(A))

        stages = [
            dict(tag="rigid, cranial head ROI", rigid=True, affine=False, masks=masks1, maxstep=0.05, minstep=0.001),
            dict(tag="rigid, skull ROI", rigid=True, affine=False, masks=masks2, maxstep=0.01, minstep=0.0005),
        ]
        if tt == "Rigid+Affine":
            stages.append(dict(tag="affine, skull ROI", rigid=False, affine=True, masks=masks2, maxstep=0.01, minstep=0.0005))

        prev, cli = tn, None
        for k, st in enumerate(stages):
            last = k == len(stages) - 1
            out_tn = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"v2_stage{k + 1}")
            nodes.append(out_tn)
            params = {
                "fixedVolume": fixed_preop_node.GetID(),
                "movingVolume": moving_reg.GetID(),
                "linearTransform": out_tn.GetID(),
                "initialTransform": prev.GetID(),
                "initializeTransformMode": "Off",
                "useRigid": bool(st["rigid"]),
                "useAffine": bool(st["affine"]),
                "samplingPercentage": float(sampling_percentage),
                "numberOfIterations": int(number_of_iterations),
                "maximumStepLength": float(st["maxstep"]),
                "minimumStepLength": float(st["minstep"]),
            }
            if st["masks"] is not None:
                params.update({"maskProcessingMode": "ROI",
                               "fixedBinaryVolume": st["masks"][0].GetID(),
                               "movingBinaryVolume": st["masks"][1].GetID()})
            if last:
                params.update({"outputVolume": output_transformed_postop_node.GetID(),
                               "backgroundFillValue": float(SENTINEL),
                               "interpolationMode": "BSpline"})
            cli = _run_brainsfit(params, f"stage {k + 1}/{len(stages)}: {st['tag']}")
            prev = out_tn

        Af = _transform_matrix(prev)
        R = Af[:3, :3]
        u, s, vt = np.linalg.svd(R)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(u @ vt) - 1) / 2, -1, 1))))
        _log(f"[v2] final transform: rotation {ang:.2f} deg, translation {np.linalg.norm(Af[:3, 3]):.1f} mm, "
                     f"scale {s.min():.4f}..{s.max():.4f}")
        if ang > 25 or abs(s.max() - 1) > 0.05:
            _log("[v2] WARNING: unusual final transform (check this case)")
        try:
            out = slicer.util.arrayFromVolume(output_transformed_postop_node)
            q = quality_metrics(pre, out, out > SENTINEL + 1.0, sp_f)
            _log("[v2] QC skull_dice={skull_dice:.3f} coverage={coverage:.3f} bone_ncc={bone_ncc:.3f} fov={fov:.3f}".format(**q))
            if q["skull_dice"] < 0.8:
                _log("[v2] WARNING: poor skull overlap after registration (check this case)")
        except Exception as e:
            _log(f"[v2] QC failed: {e}")
        return cli
    finally:
        for n in nodes:
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass