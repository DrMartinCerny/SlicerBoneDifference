import logging
import os
import sys

import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import vtk

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import binarize
import register


#
# BoneDifference
#

class BoneDifference(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Bone Difference"
        parent.categories = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["Martin Cerny, MD, PhD"]
        parent.helpText = """
Detect bone defects by subtracting preoperative and postoperative CT scans
and binarizing the positive bone-loss signal inside skull bone.

Optionally runs BRAINSFit affine registration (postop → preop) and uses the transformed postop
for the diff computation.
"""
        parent.acknowledgementText = ""


#
# BoneDifferenceWidget
#

class BoneDifferenceWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        form = qt.QFormLayout()

        # Inputs
        self.preopSelector = slicer.qMRMLNodeComboBox()
        self.preopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.preopSelector.noneEnabled = True
        self.preopSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Preop volume:", self.preopSelector)

        self.postopSelector = slicer.qMRMLNodeComboBox()
        self.postopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.postopSelector.noneEnabled = True
        self.postopSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Postop volume:", self.postopSelector)

        # --- Registration section ---
        self.registerCheckbox = qt.QCheckBox("Run affine registration (postop → preop) using BRAINSFit")
        self.registerCheckbox.checked = False
        form.addRow("", self.registerCheckbox)

        self.transformedSelector = slicer.qMRMLNodeComboBox()
        self.transformedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.transformedSelector.noneEnabled = True
        self.transformedSelector.addEnabled = True
        self.transformedSelector.renameEnabled = True
        self.transformedSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Transformed postop output:", self.transformedSelector)

        # Outputs
        self.diffOutputSelector = slicer.qMRMLNodeComboBox()
        self.diffOutputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.diffOutputSelector.noneEnabled = True
        self.diffOutputSelector.addEnabled = True
        self.diffOutputSelector.renameEnabled = True
        self.diffOutputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Diff output:", self.diffOutputSelector)

        self.maskOutputSelector = slicer.qMRMLNodeComboBox()
        self.maskOutputSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.maskOutputSelector.noneEnabled = True
        self.maskOutputSelector.addEnabled = True
        self.maskOutputSelector.renameEnabled = True
        self.maskOutputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Mask output (labelmap):", self.maskOutputSelector)

        # --- Binarization parameters ---
        # Delta HU threshold (HU)
        self.deltaThresholdSpin = qt.QSpinBox()
        self.deltaThresholdSpin.minimum = 0
        self.deltaThresholdSpin.maximum = 2000
        self.deltaThresholdSpin.singleStep = 10
        self.deltaThresholdSpin.value = 700
        form.addRow("Delta HU threshold:", self.deltaThresholdSpin)

        # Bone threshold (HU)
        self.boneThresholdSpin = qt.QDoubleSpinBox()
        self.boneThresholdSpin.minimum = -2000.0
        self.boneThresholdSpin.maximum = 4000.0
        self.boneThresholdSpin.decimals = 1
        self.boneThresholdSpin.singleStep = 10.0
        self.boneThresholdSpin.value = 300.0
        form.addRow("Bone threshold (HU):", self.boneThresholdSpin)

        # Min component size
        self.minCompSpin = qt.QSpinBox()
        self.minCompSpin.minimum = 0
        self.minCompSpin.maximum = 10_000_000
        self.minCompSpin.singleStep = 10
        self.minCompSpin.value = 50
        form.addRow("Min component size (vox):", self.minCompSpin)

        # Head cap depth
        self.headCapSpin = qt.QDoubleSpinBox()
        self.headCapSpin.minimum = 0.0
        self.headCapSpin.maximum = 500.0
        self.headCapSpin.decimals = 1
        self.headCapSpin.singleStep = 5.0
        self.headCapSpin.value = 100.0
        form.addRow("Head cap depth (mm):", self.headCapSpin)

        # Keep largest bone component
        self.keepLargestBoneCheckbox = qt.QCheckBox("Keep only largest bone component (skull)")
        self.keepLargestBoneCheckbox.checked = True
        form.addRow("", self.keepLargestBoneCheckbox)

        # Run
        self.runButton = qt.QPushButton("Run")
        form.addRow(self.runButton)

        self.layout.addLayout(form)
        self.layout.addStretch(1)

        self.logic = BoneDifferenceLogic()

        self.runButton.connect("clicked()", self.onRun)
        self.registerCheckbox.connect("toggled(bool)", self._onRegisterToggled)

        self._onRegisterToggled(self.registerCheckbox.checked)

    def _onRegisterToggled(self, checked: bool):
        self.transformedSelector.enabled = bool(checked)

    def onRun(self):
        preop = self.preopSelector.currentNode()
        postop = self.postopSelector.currentNode()

        if preop is None or postop is None:
            slicer.util.errorDisplay("Select both preop and postop volumes.")
            return

        doRegister = bool(self.registerCheckbox.checked)
        transformedNode = self.transformedSelector.currentNode() if doRegister else None

        if doRegister and transformedNode is None:
            transformedNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", "postop_transformed_to_preop"
            )
            self.transformedSelector.setCurrentNode(transformedNode)

        diffNode = self.diffOutputSelector.currentNode()
        maskNode = self.maskOutputSelector.currentNode()

        if diffNode is None:
            diffNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Diff")
            self.diffOutputSelector.setCurrentNode(diffNode)

        if maskNode is None:
            maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Mask")
            self.maskOutputSelector.setCurrentNode(maskNode)

        params = dict(
            do_registration=doRegister,
            transformed_postop_node=transformedNode,
            delta_hu_threshold=float(self.deltaThresholdSpin.value),
            bone_threshold_hu=float(self.boneThresholdSpin.value),
            min_component_size=int(self.minCompSpin.value),
            head_cap_depth_mm=float(self.headCapSpin.value),
            keep_largest_bone_component=bool(self.keepLargestBoneCheckbox.checked),
        )

        try:
            self.logic.run(preop, postop, diffNode, maskNode, **params)
        except Exception as e:
            logging.exception("BoneDifference failed")
            slicer.util.errorDisplay(f"BoneDifference failed:\n{e}")


#
# BoneDifferenceLogic
#

class BoneDifferenceLogic(ScriptedLoadableModuleLogic):

    # ---- Strict geometry checks ------------------------------------------------

    @staticmethod
    def _get_ijk_dims(volumeNode):
        img = volumeNode.GetImageData()
        if img is None:
            raise ValueError(f"Volume '{volumeNode.GetName()}' has no image data loaded.")
        return img.GetDimensions()  # (x,y,z)

    @staticmethod
    def _vtkmat4(volumeNode) -> vtk.vtkMatrix4x4:
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(m)
        return m

    @staticmethod
    def _mat4_to_numpy(m: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=np.float64)

    @staticmethod
    def _assert_same_geometry(aNode, bNode, *, tol: float = 1e-6, context: str = ""):
        """
        Strictly require identical voxel grid in RAS:
          - dimensions
          - spacing
          - origin
          - full IJK->RAS 4x4 matrix

        tol applies to float comparisons.
        """
        aName = aNode.GetName()
        bName = bNode.GetName()

        aDims = BoneDifferenceLogic._get_ijk_dims(aNode)
        bDims = BoneDifferenceLogic._get_ijk_dims(bNode)
        if tuple(aDims) != tuple(bDims):
            raise ValueError(
                f"{context}Geometry mismatch (dimensions): '{aName}' {aDims} vs '{bName}' {bDims}"
            )

        aSp = tuple(float(x) for x in aNode.GetSpacing())
        bSp = tuple(float(x) for x in bNode.GetSpacing())
        if any(abs(aSp[i] - bSp[i]) > tol for i in range(3)):
            raise ValueError(
                f"{context}Geometry mismatch (spacing): '{aName}' {aSp} vs '{bName}' {bSp}"
            )

        aOrg = tuple(float(x) for x in aNode.GetOrigin())
        bOrg = tuple(float(x) for x in bNode.GetOrigin())
        if any(abs(aOrg[i] - bOrg[i]) > tol for i in range(3)):
            raise ValueError(
                f"{context}Geometry mismatch (origin): '{aName}' {aOrg} vs '{bName}' {bOrg}"
            )

        aM = BoneDifferenceLogic._mat4_to_numpy(BoneDifferenceLogic._vtkmat4(aNode))
        bM = BoneDifferenceLogic._mat4_to_numpy(BoneDifferenceLogic._vtkmat4(bNode))
        maxAbs = float(np.max(np.abs(aM - bM)))
        if maxAbs > tol:
            raise ValueError(
                f"{context}Geometry mismatch (IJK→RAS matrix max|Δ|={maxAbs:.3g} > tol={tol}): "
                f"'{aName}' vs '{bName}'"
            )

    # ---- Direction helper used by your binarize.py ----------------------------

    @staticmethod
    def _direction_3x3_from_ijk_to_ras_matrix(volumeNode) -> np.ndarray:
        """
        Build a 3x3 direction matrix whose columns are unit direction vectors
        for I, J, K axes in RAS space (normalize columns to remove spacing scale).
        """
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(m)

        D = np.array([[m.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=np.float64)
        for c in range(3):
            n = np.linalg.norm(D[:, c])
            if n > 0:
                D[:, c] /= n
        return D

    # ---- Main pipeline ---------------------------------------------------------

    def run(
        self,
        preopNode,
        postopNode,
        diffNode,
        maskNode,
        *,
        do_registration: bool = False,
        transformed_postop_node=None,
        delta_hu_threshold: float = 700.0,
        bone_threshold_hu: float = 300.0,
        min_component_size: int = 50,
        head_cap_depth_mm: float = 100.0,
        keep_largest_bone_component: bool = True,
    ):
        logging.info("Running Bone Difference")

        # ------------------------------------------------------------------
        # Decide which postop-like volume we will subtract from preop
        # ------------------------------------------------------------------
        subtractNode = None  # will be postop or transformed_postop_node

        if do_registration:
            if transformed_postop_node is None:
                raise ValueError("do_registration=True but transformed_postop_node is None")

            logging.info("Running BRAINSFit affine registration (postop -> preop)")
            cliNode = register.register_postop_to_preop_affine(
                fixed_preop_node=preopNode,
                moving_postop_node=postopNode,
                output_transformed_postop_node=transformed_postop_node,
            )

            # Force geometry to match preop (per your earlier requirement)
            transformed_postop_node.CopyOrientation(preopNode)
            transformed_postop_node.SetSpacing(preopNode.GetSpacing())
            transformed_postop_node.SetOrigin(preopNode.GetOrigin())

            # Now enforce strict identity of geometry (this should pass)
            self._assert_same_geometry(
                preopNode,
                transformed_postop_node,
                tol=1e-6,
                context="[After BRAINSFit] ",
            )

            transformed_postop_node.CreateDefaultDisplayNodes()
            disp = transformed_postop_node.GetDisplayNode()
            if disp:
                disp.AutoWindowLevelOn()

            if cliNode:
                logging.info(f"BRAINSFit CLI status: {cliNode.GetStatusString()}")

            subtractNode = transformed_postop_node

        else:
            # No registration: require that preop and postop already match perfectly
            self._assert_same_geometry(
                preopNode,
                postopNode,
                tol=1e-6,
                context="[No registration] ",
            )
            subtractNode = postopNode

        # ------------------------------------------------------------------
        # Diff (preop - subtractNode)
        # ------------------------------------------------------------------
        preop = slicer.util.arrayFromVolume(preopNode).astype(np.float32)           # (z,y,x)
        sub = slicer.util.arrayFromVolume(subtractNode).astype(np.float32)         # (z,y,x)

        if preop.shape != sub.shape:
            # Should never happen if geometry checks passed, but keep it explicit.
            raise ValueError(f"Array shape mismatch: preop {preop.shape} vs subtract {sub.shape}")

        diff = preop - sub

        # --- Save diff output ---
        slicer.util.updateVolumeFromArray(diffNode, diff)
        diffNode.CopyOrientation(preopNode)
        diffNode.SetSpacing(preopNode.GetSpacing())
        diffNode.SetOrigin(preopNode.GetOrigin())
        diffNode.CreateDefaultDisplayNodes()
        diffDisplay = diffNode.GetDisplayNode()
        if diffDisplay:
            diffDisplay.AutoWindowLevelOn()

        # ------------------------------------------------------------------
        # Binarize mask using numpy/scipy pipeline
        # ------------------------------------------------------------------
        spacing_xyz = preopNode.GetSpacing()  # (sx, sy, sz)
        direction_3x3 = self._direction_3x3_from_ijk_to_ras_matrix(preopNode)

        mask = binarize.binarize_diff(
            diff,
            preop,
            delta_hu_threshold=float(delta_hu_threshold),
            bone_threshold_hu=float(bone_threshold_hu),
            min_component_size=int(min_component_size),
            head_cap_depth_mm=float(head_cap_depth_mm),
            keep_largest_bone_component=bool(keep_largest_bone_component),
            closing_radius_xyz=(1, 1, 1),
            spacing_xyz=spacing_xyz,
            direction_3x3=direction_3x3,
            connectivity=1,
        )

        # --- Save mask output (labelmap 0/1) ---
        slicer.util.updateVolumeFromArray(maskNode, mask.astype(np.uint8, copy=False))
        maskNode.CopyOrientation(preopNode)
        maskNode.SetSpacing(preopNode.GetSpacing())
        maskNode.SetOrigin(preopNode.GetOrigin())
        maskNode.CreateDefaultDisplayNodes()

        logging.info("Bone Difference finished")