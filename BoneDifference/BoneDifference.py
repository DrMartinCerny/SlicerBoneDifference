import logging
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import qt


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
and thresholding the difference to generate segmentation masks.
"""
        parent.acknowledgementText = ""


#
# BoneDifferenceWidget
#

class BoneDifferenceWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        layout = qt.QFormLayout()

        #
        # Preop selector
        #
        self.preopSelector = slicer.qMRMLNodeComboBox()
        self.preopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.preopSelector.setMRMLScene(slicer.mrmlScene)
        layout.addRow("Preop volume:", self.preopSelector)

        #
        # Postop selector
        #
        self.postopSelector = slicer.qMRMLNodeComboBox()
        self.postopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.postopSelector.setMRMLScene(slicer.mrmlScene)
        layout.addRow("Postop volume:", self.postopSelector)

        #
        # Diff output
        #
        self.diffOutputSelector = slicer.qMRMLNodeComboBox()
        self.diffOutputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.diffOutputSelector.addEnabled = True
        self.diffOutputSelector.setMRMLScene(slicer.mrmlScene)
        layout.addRow("Diff output:", self.diffOutputSelector)

        #
        # Mask output
        #
        self.maskOutputSelector = slicer.qMRMLNodeComboBox()
        self.maskOutputSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.maskOutputSelector.addEnabled = True
        self.maskOutputSelector.setMRMLScene(slicer.mrmlScene)
        layout.addRow("Mask output:", self.maskOutputSelector)

        #
        # Threshold slider
        #
        self.thresholdSlider = qt.QSlider(qt.Qt.Horizontal)
        self.thresholdSlider.minimum = 0
        self.thresholdSlider.maximum = 2000
        self.thresholdSlider.value = 1000

        self.thresholdLabel = qt.QLabel(str(self.thresholdSlider.value))
        self.thresholdSlider.valueChanged.connect(
            lambda v: self.thresholdLabel.setText(str(v))
        )

        thresholdRow = qt.QHBoxLayout()
        thresholdRow.addWidget(self.thresholdSlider)
        thresholdRow.addWidget(self.thresholdLabel)

        layout.addRow("Threshold:", thresholdRow)

        #
        # Run button
        #
        self.runButton = qt.QPushButton("Run")
        layout.addRow(self.runButton)

        self.layout.addLayout(layout)
        self.layout.addStretch(1)

        self.runButton.connect("clicked()", self.onRun)

        self.logic = BoneDifferenceLogic()

    def onRun(self):

        preop = self.preopSelector.currentNode()
        postop = self.postopSelector.currentNode()

        diffNode = self.diffOutputSelector.currentNode()
        maskNode = self.maskOutputSelector.currentNode()

        threshold = int(self.thresholdSlider.value)

        if not diffNode:
            diffNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", "Diff")
            self.diffOutputSelector.setCurrentNode(diffNode)

        if not maskNode:
            maskNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "Mask")
            self.maskOutputSelector.setCurrentNode(maskNode)

        self.logic.run(preop, postop, diffNode, maskNode, threshold)


#
# BoneDifferenceLogic
#

class BoneDifferenceLogic(ScriptedLoadableModuleLogic):

    def run(self, preopNode, postopNode, diffNode, maskNode, threshold):

        logging.info("Running BoneDifference")

        preop = slicer.util.arrayFromVolume(preopNode).astype(np.float32)
        postop = slicer.util.arrayFromVolume(postopNode).astype(np.float32)

        diff = preop - postop
        mask = (diff > float(threshold)).astype(np.uint8)

        #
        # Diff output
        #
        slicer.util.updateVolumeFromArray(diffNode, diff)
        diffNode.CopyOrientation(preopNode)
        diffNode.CreateDefaultDisplayNodes()

        #
        # Mask output
        #
        slicer.util.updateVolumeFromArray(maskNode, mask)
        maskNode.CopyOrientation(preopNode)
        maskNode.CreateDefaultDisplayNodes()

        logging.info("BoneDifference finished")