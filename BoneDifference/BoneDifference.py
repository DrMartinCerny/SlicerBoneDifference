import logging
import os
import sys

import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import vtk

# Make sure we can import sibling files (binarize.py) even without packaging
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import binarize


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
Assumes volumes are already registered/aligned.
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
        self.runButton = qt.QPushButton("Run (diff + binarize mask)")
        form.addRow(self.runButton)

        self.layout.addLayout(form)
        self.layout.addStretch(1)

        self.runButton.connect("clicked()", self.onRun)

        self.logic = BoneDifferenceLogic()

    def onRun(self):
        preop = self.preopSelector.currentNode()
        postop = self.postopSelector.currentNode()

        if preop is None or postop is None:
            slicer.util.errorDisplay("Select both preop and postop volumes.")
            return

        diffNode = self.diffOutputSelector.currentNode()
        maskNode = self.maskOutputSelector.currentNode()

        if diffNode is None:
            diffNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Diff")
            self.diffOutputSelector.setCurrentNode(diffNode)

        if maskNode is None:
            maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Mask")
            self.maskOutputSelector.setCurrentNode(maskNode)

        params = dict(
            delta_hu_threshold=float(self.deltaThresholdSpin.value),
            bone_threshold_hu=float(self.boneThresholdSpin.value),
            min_component_size=int(self.minCompSpin.value),
            head_cap_depth_mm=float(self.headCapSpin.value),
            keep_largest_bone_component=bool(self.keepLargestBoneCheckbox.checked),
        )

        self.logic.run(preop, postop, diffNode, maskNode, **params)


#
# BoneDifferenceLogic
#

class BoneDifferenceLogic(ScriptedLoadableModuleLogic):

    @staticmethod
    def _direction_3x3_from_ijk_to_ras_matrix(volumeNode) -> np.ndarray:
        """
        Build a 3x3 direction matrix whose columns are unit direction vectors
        for I, J, K axes in RAS space.

        We intentionally normalize columns to remove spacing scale.
        """
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(m)

        # Extract 3x3 (includes spacing scale); normalize each column
        D = np.array([[m.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=np.float64)

        for c in range(3):
            n = np.linalg.norm(D[:, c])
            if n > 0:
                D[:, c] /= n

        return D

    def run(
        self,
        preopNode,
        postopNode,
        diffNode,
        maskNode,
        *,
        delta_hu_threshold: float,
        bone_threshold_hu: float = 300.0,
        min_component_size: int = 50,
        head_cap_depth_mm: float = 100.0,
        keep_largest_bone_component: bool = True,
    ):
        logging.info("Running Bone Difference")

        preop = slicer.util.arrayFromVolume(preopNode).astype(np.float32)   # (z,y,x)
        postop = slicer.util.arrayFromVolume(postopNode).astype(np.float32) # (z,y,x)

        diff = preop - postop

        # --- Save diff output ---
        slicer.util.updateVolumeFromArray(diffNode, diff)
        diffNode.CopyOrientation(preopNode)
        diffNode.CreateDefaultDisplayNodes()
        diffDisplay = diffNode.GetDisplayNode()
        if diffDisplay:
            diffDisplay.AutoWindowLevelOn()

        # --- Binarize mask using numpy/scipy pipeline ---
        spacing_xyz = preopNode.GetSpacing()  # (sx, sy, sz)
        direction_3x3 = self._direction_3x3_from_ijk_to_ras_matrix(preopNode)  # 3x3, columns are I/J/K axes

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
            connectivity=1,  # 6-neighborhood, matches your conservative behavior
        )

        # --- Save mask output (labelmap 0/1) ---
        slicer.util.updateVolumeFromArray(maskNode, mask.astype(np.uint8, copy=False))
        maskNode.CopyOrientation(preopNode)
        maskNode.CreateDefaultDisplayNodes()

        logging.info("Bone Difference finished")